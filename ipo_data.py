import pandas as pd
import requests
import json
import datetime
import streamlit as st
from datetime import datetime, timedelta

# Cache IPO data to avoid repeated API calls
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour with no spinner
def fetch_us_ipo_data():
    """
    Fetch upcoming and recent US IPO data with improved state persistence
    """
    try:
        # Using Financial Modeling Prep API (free tier) for demonstration
        # In production, you would use a paid API with better data
        url = "https://financialmodelingprep.com/api/v3/ipo-calendar"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            # Clean and format the data
            if not df.empty:
                # Convert date strings to datetime
                df['date'] = pd.to_datetime(df['date'])
                df['expectedDate'] = pd.to_datetime(df['expectedDate'])
                
                # Sort by expected date (most recent first)
                df = df.sort_values('expectedDate', ascending=False)
                
                # Add status column
                now = datetime.now().date()
                df['status'] = df['expectedDate'].apply(lambda x: 'Upcoming' if x.date() > now else 'Listed')
                
                # Add metadata for tracking
                df.attrs['_last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                df.attrs['_data_source'] = 'API'
                
                return df
            else:
                return _get_sample_us_ipo_data()
        else:
            # Return sample data if API fails
            return _get_sample_us_ipo_data()
    except Exception as e:
        st.error(f"Error fetching US IPO data: {e}")
        # Return sample data if API fails
        return _get_sample_us_ipo_data()

def _get_sample_us_ipo_data():
    """Helper function to generate sample US IPO data with consistent timestamps"""
    # Use a fixed seed date to ensure consistency across refreshes
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    df = pd.DataFrame({
        'company': ['Sample Company 1', 'Sample Company 2'],
        'symbol': ['SMPL1', 'SMPL2'],
        'price': ['$18-$20', '$22-$25'],
        'shares': ['10M', '15M'],
        'expectedDate': [base_date + timedelta(days=5), base_date + timedelta(days=10)],
        'exchange': ['NASDAQ', 'NYSE'],
        'status': ['Upcoming', 'Upcoming']
    })
    
    # Add metadata for tracking
    df.attrs['_last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df.attrs['_data_source'] = 'Sample'
    
    return df

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour with no spinner
def fetch_indian_ipo_data():
    """
    Fetch upcoming and recent Indian IPO data with improved state persistence
    """
    try:
        # In a real implementation, you would use a paid API for Indian IPO data
        # For demonstration, we'll return sample data
        now = datetime.now()
        
        # Sample data for Indian IPOs
        data = {
            'company': [
                'Bajaj Housing Finance', 'Emcure Pharmaceuticals', 'Ola Electric', 
                'Swiggy', 'FirstCry', 'Aadhar Housing Finance', 'Waaree Energies'
            ],
            'symbol': ['BAJAJHFL', 'EMCURE', 'OLAELEC', 'SWIGGY', 'FIRSTCRY', 'AADHAR', 'WAAREE'],
            'price': [
                '₹660-₹700', '₹900-₹950', '₹430-₹460', 
                '₹350-₹370', '₹440-₹465', '₹300-₹315', '₹1,130-₹1,180'
            ],
            'issue_size': [
                '₹7,000 Cr', '₹2,500 Cr', '₹5,500 Cr', 
                '₹3,750 Cr', '₹3,300 Cr', '₹1,500 Cr', '₹3,000 Cr'
            ],
            'expectedDate': [
                now + timedelta(days=3), now + timedelta(days=7), now + timedelta(days=12),
                now + timedelta(days=15), now + timedelta(days=20), now - timedelta(days=5), now - timedelta(days=10)
            ],
            'subscription': [
                'Opening Soon', 'Opening Soon', 'Opening Soon',
                'Opening Soon', 'Opening Soon', '42.5x', '38.2x'
            ],
            'listing_gains': [
                'N/A', 'N/A', 'N/A',
                'N/A', 'N/A', '+15.2%', '+22.7%'
            ],
            'exchange': ['NSE/BSE', 'NSE/BSE', 'NSE/BSE', 'NSE/BSE', 'NSE/BSE', 'NSE/BSE', 'NSE/BSE']
        }
        
        df = pd.DataFrame(data)
        
        # Add status column
        now_date = now.date()
        df['status'] = df['expectedDate'].apply(lambda x: 'Upcoming' if x.date() > now_date else 'Listed')
        
        # Add metadata for tracking
        df.attrs['_last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df.attrs['_data_source'] = 'Sample'
        
        return df
    except Exception as e:
        st.error(f"Error fetching Indian IPO data: {e}")
        # Return minimal sample data if function fails
        return _get_sample_indian_ipo_data()

def _get_sample_indian_ipo_data():
    """Helper function to generate sample Indian IPO data with consistent timestamps"""
    # Use a fixed seed date to ensure consistency across refreshes
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    df = pd.DataFrame({
        'company': ['Sample Indian Co 1', 'Sample Indian Co 2'],
        'symbol': ['SMPIN1', 'SMPIN2'],
        'price': ['₹300-₹320', '₹500-₹550'],
        'issue_size': ['₹1,500 Cr', '₹2,200 Cr'],
        'expectedDate': [base_date + timedelta(days=5), base_date + timedelta(days=10)],
        'subscription': ['Opening Soon', 'Opening Soon'],
        'listing_gains': ['N/A', 'N/A'],
        'exchange': ['NSE/BSE', 'NSE/BSE'],
        'status': ['Upcoming', 'Upcoming']
    })
    
    # Add metadata for tracking
    df.attrs['_last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df.attrs['_data_source'] = 'Sample'
    
    return df

def format_ipo_table(df, market_type):
    """
    Format IPO data for display in Streamlit with improved state handling
    """
    # Create a copy to avoid modifying the original dataframe
    display_df = df.copy()
    
    # Preserve metadata attributes if they exist
    if hasattr(df, 'attrs'):
        for key, value in df.attrs.items():
            display_df.attrs[key] = value
    
    # Format date columns with error handling
    try:
        display_df['expectedDate'] = display_df['expectedDate'].dt.strftime('%d %b %Y')
    except Exception as e:
        # Handle case where expectedDate might not be in datetime format
        st.warning(f"Error formatting dates: {e}")
        # Try to convert to datetime if possible
        try:
            display_df['expectedDate'] = pd.to_datetime(display_df['expectedDate']).dt.strftime('%d %b %Y')
        except:
            # If all else fails, leave as is
            pass
    
    # Select and rename columns based on market type
    try:
        if market_type == 'us':
            display_df = display_df[['company', 'symbol', 'price', 'shares', 'expectedDate', 'exchange', 'status']]
            display_df.columns = ['Company', 'Symbol', 'Price Range', 'Shares Offered', 'Expected Date', 'Exchange', 'Status']
        else:  # Indian IPOs
            display_df = display_df[['company', 'symbol', 'price', 'issue_size', 'expectedDate', 'subscription', 'listing_gains', 'status']]
            display_df.columns = ['Company', 'Symbol', 'Price Range', 'Issue Size', 'Date', 'Subscription', 'Listing Gains', 'Status']
    except KeyError as e:
        st.error(f"Missing expected column in IPO data: {e}")
        # Return a minimal dataframe to prevent display errors
        if market_type == 'us':
            return pd.DataFrame(columns=['Company', 'Symbol', 'Price Range', 'Shares Offered', 'Expected Date', 'Exchange', 'Status'])
        else:
            return pd.DataFrame(columns=['Company', 'Symbol', 'Price Range', 'Issue Size', 'Date', 'Subscription', 'Listing Gains', 'Status'])
    
    return display_df

def render_ipo_section(market_type):
    """
    Render IPO section for either US or Indian market with improved stability and persistent state
    """
    # Initialize session state keys if they don't exist
    # Use a unique key for each market type to ensure proper state persistence
    session_key_data = f"{market_type}_data"
    session_key_filter = f"{market_type}_status_filter"
    session_key_last_view = f"{market_type}_last_view"
    
    # Record the current timestamp to track when this view was last rendered
    current_time = datetime.now()
    st.session_state[session_key_last_view] = current_time
    
    # Initialize data with better error handling
    if session_key_data not in st.session_state:
        try:
            if market_type == 'us':
                st.session_state[session_key_data] = fetch_us_ipo_data()
            else:  # Indian IPOs
                st.session_state[session_key_data] = fetch_indian_ipo_data()
        except Exception as e:
            st.error(f"Error initializing {market_type} IPO data: {e}")
            # Create empty DataFrame with expected columns to prevent further errors
            st.session_state[session_key_data] = pd.DataFrame({
                'company': [], 'symbol': [], 'price': [], 
                'expectedDate': [], 'status': []
            })
    
    # Set default filter if not in session state
    if session_key_filter not in st.session_state:
        st.session_state[session_key_filter] = "All"
    
    # Display title with consistent styling
    title = "US IPO Calendar" if market_type == 'us' else "Indian IPO Calendar"
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<h2 class="card-title">{title}</h2>', unsafe_allow_html=True)
    
    # Create filter options with fixed column widths
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Use session state for the radio button to maintain state across refreshes
        status_filter = st.radio(
            "Filter by Status:",
            ["All", "Upcoming", "Listed"],
            key=f"{market_type}_status_filter",
            help="Filter IPOs by their current status"
        )
    
    # Get data from session state
    data = st.session_state[f"{market_type}_data"]
    
    # Apply filters
    filtered_data = data.copy()
    if status_filter != "All":
        filtered_data = filtered_data[filtered_data['status'] == status_filter]
    
    # Format data for display
    display_df = format_ipo_table(filtered_data, market_type)
    
    # Display the table with optimized configuration and fixed widths
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Company": st.column_config.TextColumn("Company", width=200),
            "Symbol": st.column_config.TextColumn("Symbol", width=80),
            "Price Range": st.column_config.TextColumn("Price Range", width=120),
            "Shares Offered": st.column_config.TextColumn("Shares Offered", width=120),
            "Expected Date": st.column_config.TextColumn("Expected Date", width=120),
            "Exchange": st.column_config.TextColumn("Exchange", width=100),
            "Issue Size": st.column_config.TextColumn("Issue Size", width=120),
            "Date": st.column_config.TextColumn("Date", width=120),
            "Subscription": st.column_config.TextColumn("Subscription", width=120),
            "Listing Gains": st.column_config.TextColumn("Listing Gains", width=120),
            "Status": st.column_config.TextColumn("Status", width=90)
        }
    )
    
    # Add information about the data with timestamp for debugging
    last_updated = st.session_state[session_key_data].get('_last_updated', 'Unknown')
    st.markdown(f"<div class='alert alert-info'>Showing {len(display_df)} {status_filter.lower() if status_filter != 'All' else ''} IPOs. Data refreshes hourly.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a hidden element with timestamp to help with page state detection
    st.markdown(f"<div id='{market_type}-state-marker' style='display:none;' data-timestamp='{current_time}'></div>", unsafe_allow_html=True)
    
    # Add a callback to handle page reloads and maintain state
    if 'page_reload_handler' not in st.session_state:
        st.session_state.page_reload_handler = True
        
        # Create a JavaScript callback to handle page reloads
        st.markdown("""
        <script>
        // Store the current filter state before page reload
        window.addEventListener('beforeunload', function() {
            const marketTypes = ['us', 'indian'];
            marketTypes.forEach(type => {
                const filterValue = document.querySelector(`input[name="${type}_status_filter"]:checked`)?.value;
                if (filterValue) {
                    localStorage.setItem(`${type}_filter_state`, filterValue);
                }
            });
        });
        
        // Restore state after page loads
        window.addEventListener('DOMContentLoaded', function() {
            const marketTypes = ['us', 'indian'];
            marketTypes.forEach(type => {
                const savedFilter = localStorage.getItem(`${type}_filter_state`);
                if (savedFilter) {
                    // Find the radio button with this value and click it
                    const radioButton = document.querySelector(`input[name="${type}_status_filter"][value="${savedFilter}"]`);
                    if (radioButton) {
                        radioButton.click();
                    }
                }
            });
        });
        </script>
        """, unsafe_allow_html=True)