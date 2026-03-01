// static/js/main.js

// Flash message auto-hide
document.addEventListener('DOMContentLoaded', function() {
    // Auto-hide flash messages after 5 seconds
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(message => {
        setTimeout(() => {
            message.style.animation = 'slideOut 0.3s';
            setTimeout(() => {
                message.remove();
            }, 300);
        }, 5000);
    });

    // Add slide-out animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
});

// Form validation
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return true;

    const inputs = form.querySelectorAll('input[required], textarea[required], select[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.style.borderColor = '#ff4757';
            isValid = false;
            
            // Add error message
            let errorMsg = input.parentNode.querySelector('.error-message');
            if (!errorMsg) {
                errorMsg = document.createElement('span');
                errorMsg.className = 'error-message';
                errorMsg.style.color = '#ff4757';
                errorMsg.style.fontSize = '0.85rem';
                errorMsg.style.marginTop = '0.3rem';
                input.parentNode.appendChild(errorMsg);
            }
            errorMsg.textContent = 'This field is required';
        } else {
            input.style.borderColor = '#e0e0e0';
            const errorMsg = input.parentNode.querySelector('.error-message');
            if (errorMsg) errorMsg.remove();
        }
    });

    return isValid;
}

// Preview ticket before submission
function previewTicket() {
    const description = document.getElementById('description').value;
    if (!description.trim()) {
        alert('Please enter a description');
        return false;
    }

    // Show loading spinner
    const submitBtn = document.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Processing...';

    return true;
}

// Copy ticket details to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(() => {
        showNotification('Failed to copy', 'error');
    });
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `flash-message flash-${type}`;
    notification.innerHTML = `
        ${message}
        <button class="flash-close" onclick="this.parentElement.remove()">×</button>
    `;

    const container = document.querySelector('.flash-messages') || createFlashContainer();
    container.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Create flash container if it doesn't exist
function createFlashContainer() {
    const container = document.createElement('div');
    container.className = 'flash-messages';
    document.body.appendChild(container);
    return container;
}

// Filter tickets by priority
function filterTickets(priority) {
    const rows = document.querySelectorAll('.ticket-row');
    rows.forEach(row => {
        const rowPriority = row.querySelector('.priority-badge')?.textContent.toLowerCase();
        if (priority === 'all' || rowPriority === priority) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

// Search tickets
function searchTickets(searchTerm) {
    const rows = document.querySelectorAll('.ticket-row');
    searchTerm = searchTerm.toLowerCase();

    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        if (text.includes(searchTerm)) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

// Export tickets to CSV
function exportTickets() {
    const table = document.querySelector('.ticket-table');
    if (!table) return;

    const rows = table.querySelectorAll('tr');
    let csv = [];

    rows.forEach(row => {
        const cols = row.querySelectorAll('td, th');
        const rowData = [];
        cols.forEach(col => rowData.push(col.textContent.trim()));
        csv.push(rowData.join(','));
    });

    const csvContent = csv.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tickets.csv';
    a.click();
    window.URL.revokeObjectURL(url);
}