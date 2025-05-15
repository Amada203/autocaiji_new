// 全局变量
let currentPlan = null;
let skuMetadata = {};
let currentPage = 1;
const pageSize = 20;

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 检查登录状态
    checkAuthStatus();
    
    // 加载今日采样计划
    loadTodayPlan();
    
    // 绑定事件
    document.getElementById('refreshBtn')?.addEventListener('click', loadTodayPlan);
    document.getElementById('generateBtn')?.addEventListener('click', showGenerateModal);
    document.getElementById('exportBtn')?.addEventListener('click', exportPlan);
    document.getElementById('generateSubmit')?.addEventListener('click', generateNewPlan);
    
    // 捕捉率滑块事件
    const captureRateSlider = document.getElementById('targetCaptureRate');
    const captureRateValue = document.getElementById('captureRateValue');
    
    if (captureRateSlider && captureRateValue) {
        captureRateSlider.addEventListener('input', function() {
            captureRateValue.textContent = Math.round(captureRateSlider.value * 100) + '%';
        });
    }
});

// 加载今日采样计划
async function loadTodayPlan() {
    try {
        const response = await fetch('/api/sampling-plans/today', {
            headers: getAuthHeaders()
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                showLoginModal();
                return;
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        currentPlan = await response.json();
        displayPlanOverview(currentPlan);
        
        // 加载SKU元数据
        await loadSkuMetadata(currentPlan.sku_ids);
        
        // 显示SKU列表
        displaySkuList(currentPlan.sku_ids, 1);
    } catch (error) {
        console.error('加载采样计划失败:', error);
        alert('加载采样计划失败: ' + error.message);
    }
}

// 显示计划概览
function displayPlanOverview(plan) {
    if (!plan) return;
    
    const planDate = document.getElementById('planDate');
    const sampledCount = document.getElementById('sampledCount');
    const samplingRate = document.getElementById('samplingRate');
    const costSaving = document.getElementById('costSaving');
    
    if (planDate) planDate.textContent = formatDate(plan.date);
    if (sampledCount) sampledCount.textContent = plan.sku_ids.length;
    if (samplingRate) samplingRate.textContent = (plan.sampling_rate * 100).toFixed(2) + '%';
    if (costSaving) costSaving.textContent = (plan.estimated_cost_saving * 100).toFixed(2) + '%';
}

// 加载SKU元数据
async function loadSkuMetadata(skuIds) {
    // 实际项目中应该从API获取
    // 这里模拟数据
    skuMetadata = {};
    
    for (const skuId of skuIds) {
        const clusterId = hashCode(skuId) % 10;
        skuMetadata[skuId] = {
            cluster_id: clusterId,
            category: `Category${clusterId}`,
            price_level: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)]
        };
    }
}

// 显示SKU列表
function displaySkuList(skuIds, page) {
    currentPage = page;
    const tableBody = document.getElementById('skuTable');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    // 计算分页
    const totalPages = Math.ceil(skuIds.length / pageSize);
    const start = (page - 1) * pageSize;
    const end = Math.min(start + pageSize, skuIds.length);
    const pageSkuIds = skuIds.slice(start, end);
    
    // 生成表格行
    if (pageSkuIds.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="5" class="text-center">没有数据</td></tr>';
    } else {
        for (let i = 0; i < pageSkuIds.length; i++) {
            const skuId = pageSkuIds[i];
            const row = document.createElement('tr');
            
            // 序号
            const indexCell = document.createElement('td');
            indexCell.textContent = start + i + 1;
            row.appendChild(indexCell);
            
            // SKU ID
            const skuIdCell = document.createElement('td');
            skuIdCell.textContent = skuId;
            row.appendChild(skuIdCell);
            
            // 变动概率 (模拟数据)
            const probCell = document.createElement('td');
            const prob = Math.random().toFixed(4);
            probCell.textContent = prob;
            row.appendChild(probCell);
            
            // 类别
            const categoryCell = document.createElement('td');
            categoryCell.textContent = skuMetadata[skuId]?.category || '-';
            row.appendChild(categoryCell);
            
            // 价格档位
            const priceCell = document.createElement('td');
            priceCell.textContent = skuMetadata[skuId]?.price_level || '-';
            row.appendChild(priceCell);
            
            tableBody.appendChild(row);
        }
    }
    
    // 更新分页控件
    updatePagination(totalPages, page);
}

// 更新分页控件
function updatePagination(totalPages, currentPage) {
    const pagination = document.getElementById('pagination');
    if (!pagination) return;
    
    pagination.innerHTML = '';
    
    // 上一页
    const prevItem = document.createElement('li');
    prevItem.className = `page-item ${currentPage === 1 ? 'disabled' : ''}`;
    const prevLink = document.createElement('a');
    prevLink.className = 'page-link';
    prevLink.href = '#';
    prevLink.textContent = '上一页';
    prevLink.addEventListener('click', function(e) {
        e.preventDefault();
        if (currentPage > 1) {
            displaySkuList(currentPlan.sku_ids, currentPage - 1);
        }
    });
    prevItem.appendChild(prevLink);
    pagination.appendChild(prevItem);
    
    // 页码
    const maxPages = 5;
    const startPage = Math.max(1, currentPage - Math.floor(maxPages / 2));
    const endPage = Math.min(totalPages, startPage + maxPages - 1);
    
    for (let i = startPage; i <= endPage; i++) {
        const pageItem = document.createElement('li');
        pageItem.className = `page-item ${i === currentPage ? 'active' : ''}`;
        const pageLink = document.createElement('a');
        pageLink.className = 'page-link';
        pageLink.href = '#';
        pageLink.textContent = i;
        pageLink.addEventListener('click', function(e) {
            e.preventDefault();
            displaySkuList(currentPlan.sku_ids, i);
        });
        pageItem.appendChild(pageLink);
        pagination.appendChild(pageItem);
    }
    
    // 下一页
    const nextItem = document.createElement('li');
    nextItem.className = `page-item ${currentPage === totalPages ? 'disabled' : ''}`;
    const nextLink = document.createElement('a');
    nextLink.className = 'page-link';
    nextLink.href = '#';
    nextLink.textContent = '下一页';
    nextLink.addEventListener('click', function(e) {
        e.preventDefault();
        if (currentPage < totalPages) {
            displaySkuList(currentPlan.sku_ids, currentPage + 1);
        }
    });
    nextItem.appendChild(nextLink);
    pagination.appendChild(nextItem);
}

// 显示生成计划模态框
function showGenerateModal() {
    // 设置默认日期为今天
    const today = new Date();
    const predictionDateInput = document.getElementById('predictionDate');
    if (predictionDateInput) {
        predictionDateInput.value = formatDateForInput(today);
    }
    
    // 显示模态框
    const modal = new bootstrap.Modal(document.getElementById('generateModal'));
    modal.show();
}

// 生成新计划
async function generateNewPlan() {
    try {
        const predictionDate = document.getElementById('predictionDate')?.value;
        const targetCaptureRate = document.getElementById('targetCaptureRate')?.value;
        const useStratified = document.getElementById('useStratified')?.checked;
        const maxSamples = document.getElementById('maxSamples')?.value;
        
        const requestData = {
            prediction_date: predictionDate || null,
            target_capture_rate: targetCaptureRate ? parseFloat(targetCaptureRate) : 0.95,
            use_stratified_sampling: useStratified !== undefined ? useStratified : true,
            max_samples: maxSamples ? parseInt(maxSamples) : null
        };
        
        const response = await fetch('/api/sampling-plans/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeaders()
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                showLoginModal();
                return;
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // 关闭模态框
        const modal = bootstrap.Modal.getInstance(document.getElementById('generateModal'));
        if (modal) modal.hide();
        
        // 加载新计划
        currentPlan = await response.json();
        displayPlanOverview(currentPlan);
        
        // 加载SKU元数据
        await loadSkuMetadata(currentPlan.sku_ids);
        
        // 显示SKU列表
        displaySkuList(currentPlan.sku_ids, 1);
    } catch (error) {
        console.error('生成采样计划失败:', error);
        alert('生成采样计划失败: ' + error.message);
    }
}

// 导出计划
function exportPlan() {
    if (!currentPlan) {
        alert('没有可导出的采样计划');
        return;
    }
    
    // 创建CSV内容
    let csvContent = 'data:text/csv;charset=utf-8,';
    csvContent += 'SKU ID,类别,价格档位\n';
    
    for (const skuId of currentPlan.sku_ids) {
        const metadata = skuMetadata[skuId] || {};
        csvContent += `${skuId},${metadata.category || ''},${metadata.price_level || ''}\n`;
    }
    
    // 创建下载链接
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `采样计划_${formatDateForInput(new Date(currentPlan.date))}.csv`);
    document.body.appendChild(link);
    
    // 触发下载
    link.click();
    
    // 清理
    document.body.removeChild(link);
}