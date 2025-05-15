// 全局变量
let loginModal = null;

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化模态框
    loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
    
    // 绑定登录按钮事件
    document.getElementById('loginBtn')?.addEventListener('click', showLoginModal);
    document.getElementById('loginSubmit')?.addEventListener('click', submitLogin);
    
    // 绑定退出按钮事件
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', logout);
    }
    
    // 检查登录状态
    checkAuthStatus();
});

// 显示登录模态框
function showLoginModal() {
    // 清空表单
    document.getElementById('username').value = '';
    document.getElementById('password').value = '';
    document.getElementById('loginError').classList.add('d-none');
    
    // 显示模态框
    loginModal.show();
}

// 提交登录
async function submitLogin() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    if (!username || !password) {
        showLoginError('请输入用户名和密码');
        return;
    }
    
    try {
        const response = await fetch('/api/auth/token', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                'username': username,
                'password': password
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '登录失败');
        }
        
        const data = await response.json();
        
        // 保存token
        localStorage.setItem('token', data.access_token);
        
        // 更新UI
        updateAuthUI(true, username);
        
        // 关闭模态框
        loginModal.hide();
        
        // 刷新页面数据
        if (typeof loadTodayPlan === 'function') {
            loadTodayPlan();
        }
        if (typeof loadStats === 'function') {
            loadStats();
        }
    } catch (error) {
        showLoginError(error.message);
    }
}

// 显示登录错误
function showLoginError(message) {
    const errorElement = document.getElementById('loginError');
    errorElement.textContent = message;
    errorElement.classList.remove('d-none');
}

// 退出登录
function logout() {
    // 清除token
    localStorage.removeItem('token');
    
    // 更新UI
    updateAuthUI(false);
    
    // 刷新页面
    window.location.reload();
}

// 检查认证状态
function checkAuthStatus() {
    const token = localStorage.getItem('token');
    if (token) {
        // 解析JWT获取用户名
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            updateAuthUI(true, payload.sub);
        } catch (e) {
            updateAuthUI(true);
        }
    } else {
        updateAuthUI(false);
    }
}

// 更新认证UI
function updateAuthUI(isLoggedIn, username = null) {
    const loginBtn = document.getElementById('loginBtn');
    const userInfo = document.getElementById('userInfo');
    const logoutBtn = document.getElementById('logoutBtn');
    
    if (isLoggedIn) {
        if (loginBtn) loginBtn.style.display = 'none';
        if (userInfo) {
            userInfo.style.display = 'block';
            userInfo.textContent = username || '已登录';
        }
        if (logoutBtn) logoutBtn.style.display = 'block';
    } else {
        if (loginBtn) loginBtn.style.display = 'block';
        if (userInfo) userInfo.style.display = 'none';
        if (logoutBtn) logoutBtn.style.display = 'none';
    }
}

// 获取认证头
function getAuthHeaders() {
    const token = localStorage.getItem('token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
}

// 格式化日期
function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('zh-CN');
}

// 格式化日期为输入框格式
function formatDateForInput(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

// 计算字符串哈希值（用于模拟数据）
function hashCode(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
}