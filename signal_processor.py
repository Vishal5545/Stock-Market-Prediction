import pandas as pd
from typing import List, Union, Any

def process_trading_signal_reasons(reasons: Union[str, List[str], pd.Series, Any]) -> List[str]:
    """
    Process trading signal reasons into a consistent list format.
    
    This function handles various input types (string, list, Series) and formats them
    into a clean list of individual reasons, with proper splitting of comma-separated values.
    
    Args:
        reasons: The trading signal reasons in various possible formats
                - String (possibly comma-separated)
                - List of strings
                - Pandas Series
                - Any other type (will be converted to string)
    
    Returns:
        List[str]: A list of individual reason strings, properly formatted
    """
    # Handle empty or None input
    if reasons is None:
        return ["No specific reasons available"]
    
    # Convert to list based on input type
    if isinstance(reasons, list):
        # Already a list, but ensure all items are strings
        reasons_list = [str(item) for item in reasons]
    elif isinstance(reasons, pd.Series):
        # Handle pandas Series - take the last value if not empty
        if reasons.empty:
            reasons_list = ["No specific reasons"]
        else:
            # Get the last value and ensure it's a string
            last_value = reasons.iloc[-1]
            if isinstance(last_value, list):
                # Handle case where Series contains a list
                reasons_list = [str(item) for item in last_value]
            else:
                reasons_list = [str(last_value)]
    else:
        # Convert any other type to string and make it a single-item list
        reasons_list = [str(reasons)]
    
    # Process comma-separated strings
    processed_list = []
    for item in reasons_list:
        if ',' in item:
            # Split by comma and add each part as a separate reason
            processed_list.extend([r.strip() for r in item.split(',')])
        else:
            processed_list.append(item.strip())
    
    # Filter out empty strings
    processed_list = [reason for reason in processed_list if reason]
    
    # If we end up with an empty list, provide a default reason
    if not processed_list:
        processed_list = ["No specific reasons available"]
    
    return processed_list


def get_signal_display_class(recommendation: str) -> str:
    """
    Get the appropriate CSS class for displaying a trading signal.
    
    Args:
        recommendation: The trading signal recommendation (Buy, Sell, or Neutral)
    
    Returns:
        str: The CSS class name for styling the signal display
    """
    recommendation = str(recommendation).strip().lower()
    
    if recommendation == "buy":
        return "signal-box-buy"
    elif recommendation == "sell":
        return "signal-box-sell"
    else:
        return "signal-box-neutral"