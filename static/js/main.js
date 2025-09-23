// Cuneiform Satır Bölme Sistemi - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // File upload drag and drop functionality
    const fileInput = document.getElementById('file');
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');

    if (fileInput && uploadForm) {
        // Drag and drop events
        const uploadArea = fileInput.closest('.form-control');
        
        if (uploadArea) {
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    updateFileDisplay();
                }
            });
        }

        // File input change event
        fileInput.addEventListener('change', function() {
            updateFileDisplay();
        });

        // Form submit event
        uploadForm.addEventListener('submit', function(e) {
            if (fileInput.files.length === 0) {
                e.preventDefault();
                showAlert('Lütfen bir dosya seçin!', 'warning');
                return;
            }

            // Validate file size (16MB max)
            const file = fileInput.files[0];
            const maxSize = 16 * 1024 * 1024; // 16MB in bytes
            
            if (file.size > maxSize) {
                e.preventDefault();
                showAlert('Dosya boyutu 16MB\'dan büyük olamaz!', 'danger');
                return;
            }

            // Show loading state
            showLoadingState();
        });
    }

    // Download progress indicators
    document.querySelectorAll('a[href*="download"]').forEach(link => {
        link.addEventListener('click', function() {
            const originalText = this.innerHTML;
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> İndiriliyor...';
            
            // Reset after 3 seconds
            setTimeout(() => {
                this.innerHTML = originalText;
            }, 3000);
        });
    });

    // Auto-refresh for results page
    if (window.location.pathname.includes('/results/')) {
        // Refresh every 30 seconds to check for new results
        setTimeout(function() {
            location.reload();
        }, 30000);
    }
});

// Update file display when file is selected
function updateFileDisplay() {
    const fileInput = document.getElementById('file');
    const uploadBtn = document.getElementById('uploadBtn');
    
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const fileSize = formatFileSize(file.size);
        
        // Update button text
        if (uploadBtn) {
            uploadBtn.innerHTML = `<i class="fas fa-magic"></i> ${file.name} (${fileSize}) - İşlemi Başlat`;
        }
        
        // Show file preview
        showFilePreview(file);
    }
}

// Show file preview
function showFilePreview(file) {
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            // Remove existing preview
            const existingPreview = document.getElementById('filePreview');
            if (existingPreview) {
                existingPreview.remove();
            }
            
            // Create preview
            const preview = document.createElement('div');
            preview.id = 'filePreview';
            preview.className = 'mt-3';
            preview.innerHTML = `
                <div class="card">
                    <div class="card-body text-center">
                        <h6>Seçilen Dosya Önizlemesi</h6>
                        <img src="${e.target.result}" class="img-fluid rounded" style="max-height: 200px;">
                        <p class="mt-2 text-muted">${file.name} (${formatFileSize(file.size)})</p>
                    </div>
                </div>
            `;
            
            // Insert after file input
            const fileInput = document.getElementById('file');
            fileInput.parentNode.insertBefore(preview, fileInput.parentNode.nextSibling);
        };
        reader.readAsDataURL(file);
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Show loading state
function showLoadingState() {
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) {
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> İşleniyor...';
        uploadBtn.classList.add('loading');
    }
    
    // Show progress bar
    showProgressBar();
}

// Show progress bar
function showProgressBar() {
    const existingProgress = document.getElementById('uploadProgress');
    if (existingProgress) {
        existingProgress.remove();
    }
    
    const progress = document.createElement('div');
    progress.id = 'uploadProgress';
    progress.className = 'mt-3';
    progress.innerHTML = `
        <div class="alert alert-info">
            <i class="fas fa-spinner fa-spin"></i>
            <strong>İşlem devam ediyor...</strong>
            <div class="progress mt-2">
                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                     role="progressbar" style="width: 100%"></div>
            </div>
            <small class="text-muted">Bu işlem birkaç dakika sürebilir. Lütfen bekleyin...</small>
        </div>
    `;
    
    // Insert after form
    const form = document.getElementById('uploadForm');
    form.parentNode.insertBefore(progress, form.nextSibling);
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of main content
    const main = document.querySelector('main');
    main.insertBefore(alertDiv, main.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Copy to clipboard functionality
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showAlert('Panoya kopyalandı!', 'success');
    }, function(err) {
        showAlert('Kopyalama başarısız!', 'danger');
    });
}

// Download all files as ZIP
function downloadAll(timestamp) {
    const link = document.createElement('a');
    link.href = `/download_all/${timestamp}`;
    link.download = `cuneiform_lines_${timestamp}.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Table sorting functionality
function sortTable(columnIndex) {
    const table = document.querySelector('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    const isAscending = table.getAttribute('data-sort-direction') !== 'asc';
    
    rows.sort((a, b) => {
        const aText = a.cells[columnIndex].textContent.trim();
        const bText = b.cells[columnIndex].textContent.trim();
        
        if (isAscending) {
            return aText.localeCompare(bText);
        } else {
            return bText.localeCompare(aText);
        }
    });
    
    // Clear tbody and append sorted rows
    tbody.innerHTML = '';
    rows.forEach(row => tbody.appendChild(row));
    
    // Update sort direction
    table.setAttribute('data-sort-direction', isAscending ? 'asc' : 'desc');
}

// Search functionality for results table
function searchTable() {
    const input = document.getElementById('searchInput');
    const filter = input.value.toLowerCase();
    const table = document.querySelector('table');
    const rows = table.querySelectorAll('tbody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(filter) ? '' : 'none';
    });
}

// Initialize search if search input exists
const searchInput = document.getElementById('searchInput');
if (searchInput) {
    searchInput.addEventListener('keyup', searchTable);
}
