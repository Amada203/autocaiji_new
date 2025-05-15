// 主JavaScript文件 - 处理通用页面逻辑

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化工具提示
    initTooltips();
    
    // 初始化弹出框
    initPopovers();
    
    // 其他通用初始化逻辑
    initCommonComponents();
});

// 初始化工具提示
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// 初始化弹出框
function initPopovers() {
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// 初始化通用组件
function initCommonComponents() {
    // 可以在这里添加其他需要在所有页面初始化的组件
    console.log('通用组件初始化完成');
}

// 显示加载状态
function showLoading(element) {
    if (!element) return;
    
    const originalHTML = element.innerHTML;
    element.setAttribute('data-original-html', originalHTML);
    element.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 加载中...`;
    element.disabled = true;
}

// 隐藏加载状态
function hideLoading(element) {
    if (!element) return;
    
    const originalHTML = element.getAttribute('data-original-html');
    if (originalHTML) {
        element.innerHTML = originalHTML;
        element.removeAttribute('data-original-html');
    }
    element.disabled = false;
}

// 显示成功消息
function showSuccess(message, duration = 3000) {
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) return;
    
    const toastEl = document.createElement('div');
    toastEl.className = 'toast show align-items-center text-white bg-success';
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');
    
    toastEl.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    toastContainer.appendChild(toastEl);
    
    // 自动隐藏
    setTimeout(() => {
        toastEl.classList.remove('show');
        setTimeout(() => {
            toastEl.remove();
        }, 500);
    }, duration);
}

// 显示错误消息
function showError(message, duration = 5000) {
    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) return;
    
    const toastEl = document.createElement('div');
    toastEl.className = 'toast show align-items-center text-white bg-danger';
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');
    
    toastEl.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    toastContainer.appendChild(toastEl);
    
    // 自动隐藏
    setTimeout(() => {
        toastEl.classList.remove('show');
        setTimeout(() => {
            toastEl.remove();
        }, 500);
    }, duration);
}

// 确认对话框
async function showConfirm(message) {
    return new Promise((resolve) => {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.tabIndex = -1';
        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">确认</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p>${message}</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" id="confirmCancel">取消</button>
                        <button type="button" class="btn btn-primary" id="confirmOk">确定</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        const confirmModal = new bootstrap.Modal(modal);
        confirmModal.show();
        
        document.getElementById('confirmOk').addEventListener('click', function() {
            confirmModal.hide();
            resolve(true);
        });
        
        document.getElementById('confirmCancel').addEventListener('click', function() {
            confirmModal.hide();
            resolve(false);
        });
        
        modal.addEventListener('hidden.bs.modal', function() {
            modal.remove();
        });
    });
}