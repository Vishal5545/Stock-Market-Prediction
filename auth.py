import streamlit as st
import bcrypt
import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import datetime
import smtplib
from email.mime.text import MIMEText
import random
import re
import ssl
import socket

# Load environment variables
load_dotenv()

# Initialize MongoDB connection
try:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client['stockmarket']
    users_collection = db['users']
    users_collection.create_index([("email", pymongo.ASCENDING)], unique=True)
except Exception as e:
    st.error(f"Database connection error: {str(e)}")

# Email configuration with fallback options
EMAIL_CONFIG = {
    'primary': {
        'host': 'smtp.gmail.com',
        'port': 587,
        'use_tls': True
    },
    'fallback': {
        'host': 'smtp.gmail.com',
        'port': 465,
        'use_ssl': True
    }
}

# Debug flag for email issues
EMAIL_DEBUG = True

def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'email' not in st.session_state:
        st.session_state.email = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'otp' not in st.session_state:
        st.session_state.otp = None
    if 'otp_verified' not in st.session_state:
        st.session_state.otp_verified = False
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = "login"

def send_otp_with_retry(email):
    """Send OTP with fallback options and improved debugging"""
    # Generate OTP for any email address (restriction removed)
    otp = random.randint(100000, 999999)
    st.session_state.otp = str(otp)
    
    # Create a more detailed message with instructions
    message_text = f"""Your OTP for Stock Prediction App is: {otp}

If you didn't request this code, please ignore this email.

Note: This is an automated message, please do not reply to this email."""
    
    msg = MIMEText(message_text)
    msg['Subject'] = 'Stock Prediction App - Email Verification Code'
    sender_email = os.getenv("EMAIL_HOST_USER")
    
    # Validate sender email exists
    if not sender_email:
        return False, "Email configuration error: Sender email not found in environment variables"
    
    # Improve email headers to avoid spam filters
    msg['From'] = f"Stock Prediction App <{sender_email}>"
    msg['To'] = email
    msg['X-Priority'] = '1'  # High priority
    msg['X-MSMail-Priority'] = 'High'
    msg['Importance'] = 'High'
    
    # Add debug information if enabled
    debug_info = []
    if EMAIL_DEBUG:
        debug_info.append(f"Attempting to send email to: {email}")
        debug_info.append(f"From address: {sender_email}")
        debug_info.append(f"Using Gmail SMTP")
        # Display debug info immediately
        st.session_state.email_debug = "\n".join(debug_info)
    
    # Get password and validate it exists
    email_password = os.getenv("EMAIL_HOST_PASSWORD")
    if not email_password:
        if EMAIL_DEBUG:
            debug_info.append("Email password not found in environment variables")
            st.session_state.email_debug = "\n".join(debug_info)
        return False, "Email configuration error: Password not found in environment variables"
    
    # Try different password formats for Gmail App Password
    # Gmail App Passwords can be entered with or without spaces
    password_variants = [
        email_password,                    # Original password as-is
        email_password.strip(),           # Remove leading/trailing spaces
        email_password.replace(" ", "")   # Remove all spaces
    ]
    
    # Display password format for debugging (without revealing actual password)
    if EMAIL_DEBUG:
        debug_info.append(f"Password length: {len(email_password)} characters")
        debug_info.append(f"Password format: {'spaces detected' if ' ' in email_password else 'no spaces detected'}")
        debug_info.append("Will try multiple password formats for Gmail compatibility")
        st.session_state.email_debug = "\n".join(debug_info)
    
    # Try primary configuration first (TLS)
    primary_exception = None
    try:
        if EMAIL_DEBUG:
            debug_info.append(f"Trying primary SMTP configuration (TLS, port 587)")
            st.session_state.email_debug = "\n".join(debug_info)
        
        # Create connection with extended timeout    
        with smtplib.SMTP(EMAIL_CONFIG['primary']['host'], 
                         EMAIL_CONFIG['primary']['port'], 
                         timeout=60) as server:
            # Enable detailed debug output
            if EMAIL_DEBUG:
                server.set_debuglevel(1)
                
            # Identify ourselves to the SMTP server
            server.ehlo()
            # Secure the connection
            server.starttls()
            # Re-identify ourselves over TLS connection
            server.ehlo()
            
            # Try each password variant
            auth_success = False
            for i, pwd in enumerate(password_variants):
                try:
                    if EMAIL_DEBUG:
                        debug_info.append(f"Trying password variant {i+1}")
                        st.session_state.email_debug = "\n".join(debug_info)
                    
                    server.login(sender_email, pwd)
                    auth_success = True
                    
                    if EMAIL_DEBUG:
                        debug_info.append(f"SMTP login successful with password variant {i+1}")
                        st.session_state.email_debug = "\n".join(debug_info)
                    
                    # Send email with the working password
                    server.send_message(msg)
                    
                    if EMAIL_DEBUG:
                        debug_info.append("Primary SMTP succeeded")
                        st.session_state.email_debug = "\n".join(debug_info)
                    
                    return True, "OTP sent successfully to your email"
                except smtplib.SMTPAuthenticationError as auth_error:
                    if i == len(password_variants) - 1:  # Last attempt failed
                        if EMAIL_DEBUG:
                            debug_info.append(f"All password variants failed for TLS connection")
                            debug_info.append(f"Last error: {str(auth_error)}")
                            st.session_state.email_debug = "\n".join(debug_info)
                        # Will try fallback method
                    continue
                except Exception as e:
                    # Other exceptions should be raised immediately
                    raise
            
            # If we get here, all password variants failed
            raise smtplib.SMTPAuthenticationError(535, "Authentication failed for all password variants")
    except Exception as e:
        primary_exception = e
        if EMAIL_DEBUG:
            debug_info.append(f"Primary SMTP failed: {str(e)}")
            debug_info.append(f"Error type: {type(e).__name__}")
            st.session_state.email_debug = "\n".join(debug_info)
    
    # Try fallback configuration with SSL
    try:
        if EMAIL_DEBUG:
            debug_info.append(f"Trying fallback SMTP configuration (SSL, port 465)")
            st.session_state.email_debug = "\n".join(debug_info)
            
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(EMAIL_CONFIG['fallback']['host'],
                            EMAIL_CONFIG['fallback']['port'],
                            context=context,
                            timeout=60) as server:
            # Enable detailed debug output
            if EMAIL_DEBUG:
                server.set_debuglevel(1)
                
            # Try each password variant
            auth_success = False
            for i, pwd in enumerate(password_variants):
                try:
                    if EMAIL_DEBUG:
                        debug_info.append(f"Trying password variant {i+1} with SSL")
                        st.session_state.email_debug = "\n".join(debug_info)
                    
                    server.login(sender_email, pwd)
                    auth_success = True
                    
                    if EMAIL_DEBUG:
                        debug_info.append(f"SMTP_SSL login successful with password variant {i+1}")
                        st.session_state.email_debug = "\n".join(debug_info)
                    
                    # Send email with the working password
                    server.send_message(msg)
                    
                    if EMAIL_DEBUG:
                        debug_info.append("Fallback SMTP succeeded")
                        st.session_state.email_debug = "\n".join(debug_info)
                    
                    return True, "OTP sent successfully to your email (using fallback)"
                except smtplib.SMTPAuthenticationError as auth_error:
                    if i == len(password_variants) - 1:  # Last attempt failed
                        if EMAIL_DEBUG:
                            debug_info.append(f"All password variants failed for SSL connection")
                            debug_info.append(f"Last error: {str(auth_error)}")
                            st.session_state.email_debug = "\n".join(debug_info)
                        # Will try fallback method
                    continue
                except Exception as e:
                    # Other exceptions should be raised immediately
                    raise
            
            # If we get here, all password variants failed
            raise smtplib.SMTPAuthenticationError(535, "Authentication failed for all password variants")
    except Exception as e2:
        if EMAIL_DEBUG:
            debug_info.append(f"Fallback SMTP failed: {str(e2)}")
            debug_info.append(f"Error type: {type(e2).__name__}")
            debug_info.append("\nPossible solutions:")
            debug_info.append("1. For Gmail with 2FA enabled, you MUST use an App Password")
            debug_info.append("   Generate one at: https://myaccount.google.com/apppasswords")
            debug_info.append("2. Make sure 'Less secure app access' is enabled if not using 2FA")
            debug_info.append("3. Check if your Gmail account has SMTP access enabled")
            debug_info.append("4. Verify your .env file has the correct EMAIL_HOST_PASSWORD format")
            debug_info.append("   - App Passwords are typically 16 characters without spaces")
            debug_info.append("   - Google displays them with spaces, but they should be entered without spaces")
            debug_info.append("5. Check if your account has been temporarily blocked by Google for security reasons")
            debug_info.append("6. Try enabling 'Allow less secure apps' in your Google account settings if not using 2FA")
            debug_info.append("7. Check if you need to unlock captcha: https://accounts.google.com/DisplayUnlockCaptcha")
            st.session_state.email_debug = "\n".join(debug_info)
            
            # Add a test function to display debug info in the UI
            st.error(f"Email sending failed: {str(e2)}")
            if hasattr(st.session_state, 'email_debug'):
                st.code(st.session_state.email_debug, language="text")
            
        return False, "Failed to send OTP. Please check your email spam folder or try a different email address."

# Replace the old send_otp function with the new one
send_otp = send_otp_with_retry

def test_email_connection():
    """Test email connection and display detailed debug information"""
    debug_info = []
    debug_info.append("=== Email Configuration Test ===")
    
    # Check environment variables
    sender_email = os.getenv("EMAIL_HOST_USER")
    email_password = os.getenv("EMAIL_HOST_PASSWORD")
    
    debug_info.append(f"Sender email: {sender_email if sender_email else 'Not found'}")
    debug_info.append(f"Password present: {'Yes' if email_password else 'No'}")
    if email_password:
        # Clean up password for testing (remove whitespace)
        clean_password = email_password.strip()
        debug_info.append(f"Password length: {len(email_password)} characters")
        debug_info.append(f"Password format: {'spaces detected' if ' ' in email_password else 'no spaces detected'}")
        debug_info.append(f"Note: Gmail App Passwords should include spaces as shown in Google's interface")
    
    # Test SMTP connection
    debug_info.append("\n=== Testing SMTP Connection ===")
    try:
        debug_info.append(f"Connecting to {EMAIL_CONFIG['primary']['host']}:{EMAIL_CONFIG['primary']['port']}")
        with smtplib.SMTP(EMAIL_CONFIG['primary']['host'], EMAIL_CONFIG['primary']['port'], timeout=10) as server:
            debug_info.append("Connection established")
            server.ehlo()
            debug_info.append("EHLO successful")
            
            # Test TLS
            debug_info.append("Starting TLS...")
            server.starttls()
            debug_info.append("TLS successful")
            
            # Test authentication if credentials exist
            if sender_email and email_password:
                debug_info.append("Testing authentication...")
                try:
                    # Try with original password first
                    try:
                        server.login(sender_email, email_password)
                        debug_info.append("Authentication successful with original password!")
                    except:
                        # If that fails, try with cleaned password
                        clean_password = email_password.strip()
                        debug_info.append(f"Trying with cleaned password (spaces {'preserved' if ' ' in clean_password else 'removed'})...")
                        server.login(sender_email, clean_password)
                        debug_info.append("Authentication successful with cleaned password!")
                except Exception as e:
                    debug_info.append(f"Authentication failed: {str(e)}")
                    debug_info.append("\nIf using Gmail:")
                    debug_info.append("1. Make sure you're using an App Password if 2FA is enabled")
                    debug_info.append("2. Generate an App Password at: https://myaccount.google.com/apppasswords")
                    debug_info.append("3. Check if you need to unlock captcha: https://accounts.google.com/DisplayUnlockCaptcha")
                    debug_info.append("4. Check if your account has security restrictions")
    except Exception as e:
        debug_info.append(f"Connection failed: {str(e)}")
    
    # Try fallback SSL connection
    debug_info.append("\n=== Testing SSL Fallback Connection ===")
    try:
        debug_info.append(f"Connecting to {EMAIL_CONFIG['fallback']['host']}:{EMAIL_CONFIG['fallback']['port']} using SSL")
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(EMAIL_CONFIG['fallback']['host'], EMAIL_CONFIG['fallback']['port'], context=context, timeout=10) as server:
            debug_info.append("SSL Connection established")
            
            # Test authentication if credentials exist
            if sender_email and email_password:
                debug_info.append("Testing authentication...")
                try:
                    server.login(sender_email, email_password)
                    debug_info.append("SSL Authentication successful!")
                except Exception as e:
                    debug_info.append(f"SSL Authentication failed: {str(e)}")
    except Exception as e:
        debug_info.append(f"SSL Connection failed: {str(e)}")
    
    return "\n".join(debug_info)

def verify_otp(entered_otp):
    """Verify entered OTP"""
    if st.session_state.otp and entered_otp == st.session_state.otp:
        st.session_state.otp_verified = True
        return True
    return False

def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed_password):
    """Verify password"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def signup_user(first_name, last_name, email, password):
    """Create new user account"""
    try:
        if not all([first_name, last_name, email, password]):
            return False, "All fields are required"
        
        if not is_valid_email(email):
            return False, "Invalid email format"
            
        if len(password) < 6:
            return False, "Password must be at least 6 characters"
            
        if users_collection.find_one({"email": email}):
            return False, "Email already registered"
        
        if not st.session_state.otp_verified:
            return False, "Please verify your email with OTP"
        
        users_collection.insert_one({
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "password": hash_password(password),
            "created_at": datetime.datetime.utcnow()
        })
        return True, "Account created successfully"
    except Exception as e:
        return False, str(e)

def login_user(email, password):
    """Login user"""
    try:
        user = users_collection.find_one({"email": email})
        if user and verify_password(password, user['password']):
            # Set all required session state variables
            st.session_state.logged_in = True
            st.session_state.email = email
            st.session_state.username = user.get('first_name', email.split('@')[0])
            st.session_state.just_logged_in = True  # Flag for welcome message
            
            # Update last login time
            users_collection.update_one(
                {"email": email},
                {"$set": {"last_login": datetime.datetime.utcnow()}}
            )
            return True, "Login successful"
        return False, "Invalid email or password"
    except Exception as e:
        return False, str(e)

def logout_user():
    """Logout user"""
    try:
        st.session_state.logged_in = False
        st.session_state.email = None
        st.session_state.username = None
        st.session_state.just_logged_in = False
        return True, "Logged out successfully"
    except Exception as e:
        return False, str(e)

def show_login_page():
    """Display enhanced login/signup interface"""
    # Load custom CSS
    with open('auth.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Clear previous session state
    if not st.session_state.get('logged_in', False):
        for key in ['email', 'username', 'just_logged_in']:
            if key in st.session_state:
                del st.session_state[key]

    auth_mode = "login" if "auth_mode" not in st.session_state else st.session_state.auth_mode
    
    # Center the login form
    _, col, _ = st.columns([1,2,1])
    with col:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        
        # Auth Header
        st.markdown('<div class="auth-header"><h1>Stock Market Prediction</h1></div>', unsafe_allow_html=True)
        
        # Auth Tabs
        st.markdown(
            f'''
            <div class="auth-tabs">
                <div class="auth-tab {'active' if auth_mode == 'login' else ''}" 
                    onclick="window.location.href='?mode=login'">Login</div>
                <div class="auth-tab {'active' if auth_mode == 'signup' else ''}" 
                    onclick="window.location.href='?mode=signup'">Sign Up</div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        
        if auth_mode == "login":
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            if st.button("Sign In"):
                success, message = login_user(email, password)
                if success:
                    st.success(message)
                    # Set session state for welcome message
                    st.session_state.just_logged_in = True
                    st.rerun()
                else:
                    st.error(message)
            
            st.markdown(
                '<div class="auth-switch">New user? <a href="#" id="show-signup">Create account</a></div>',
                unsafe_allow_html=True
            )
            if st.button("Create Account", key="create_account"):
                st.session_state.auth_mode = "signup"
                st.rerun()
                
        else:  # Signup mode
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name", placeholder="John")
            with col2:
                last_name = st.text_input("Last Name", placeholder="Doe")
            
            email = st.text_input("Email", placeholder="your@email.com")
            
            if not st.session_state.get('otp_verified', False):
                # OTP verification section
                col1, col2 = st.columns([2,1])
                with col1:
                    if st.button("Send Code"):
                        if is_valid_email(email):
                            with st.spinner("Sending verification code..."):
                                success, message = send_otp(email)
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                                    # Show debug info if available
                                    if hasattr(st.session_state, 'email_debug'):
                                        with st.expander("Technical Details"):
                                            st.code(st.session_state.email_debug, language="text")
                        else:
                            st.error("Invalid email")
                with col2:
                    otp = st.text_input("Code", key="otp_input", placeholder="123456")
                    if st.button("Verify"):
                        if verify_otp(otp):
                            st.success("Verified!")
                        else:
                            st.error("Invalid")
            
            password = st.text_input("Password", type="password", placeholder="Min. 6 characters")
            confirm_password = st.text_input("Confirm", type="password", placeholder="Re-enter password")
            
            if st.button("Sign Up"):
                if password != confirm_password:
                    st.error("Passwords don't match")
                else:
                    success, message = signup_user(first_name, last_name, email, password)
                    if success:
                        st.success(message)
                        st.session_state.auth_mode = "login"
                        st.rerun()
                    else:
                        st.error(message)
            
            st.markdown(
                '<div class="auth-switch">Already have an account? <a href="#" id="show-login">Sign in</a></div>',
                unsafe_allow_html=True
            )
            if st.button("Back to Sign In", key="back_to_login"):
                st.session_state.auth_mode = "login"
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)