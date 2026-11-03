"""ai_engine/config.py
Shared taxonomy configuration updated for new_Customer_support_data.
Categories reflect ticket resolution queues, and Priorities reflect triage levels.
"""

ALLOWED_CATEGORIES = [
    'Billing and Payments',
    'Customer Service',
    'General Inquiry',
    'Human Resources',
    'IT Support',
    'Product Support',
    'Returns and Exchanges',
    'Sales and Pre-Sales',
    'Service Outages and Maintenance',
    'Technical Support',
    'Others'
]

ALLOWED_PRIORITIES = ['High', 'Medium', 'Low']
