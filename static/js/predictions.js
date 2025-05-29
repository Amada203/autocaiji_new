// 全局变量
let allPredictions = [];
let currentPage = 1;
const pageSize = 10;
let currentFilter = 0; // 默认显示所有预测
let skuDetailChart = null;
let timeDistChart = null;

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 检查登录状态
    checkAuthStatus();
    
    // 加载预测结果
    loadPredictions();
    
    // 加载预测统计
    loadPredictionStats();
    
    // 绑定事件
    document.getElementById('refreshBtn')?.addEventListener('click', function() {
        loadPredictions();
        loadPredictionStats();
    });
    
    // 绑定筛选下拉菜单事件
    const filterItems = document.querySelectorAll('.dropdown-item[data-prob]');
    filterItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            currentFilter = parseFloat(this.getAttribute('data-prob'));
            currentPage = 1;
            filterAndDisplayPredictions();
        });
    });

    const btn = document.getElementById('realtimePredictBtn');
    if (btn) {
        btn.addEventListener('click', predictRealtime);
    }
});

// 加载预测结果
async function loadPredictions() {
    try {
        const response = await fetch('/api/predictions/latest', {
            headers: getAuthHeaders()
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                showLoginModal();
                return;
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        allPredictions = await response.json();
        
        // 过滤并显示预测结果
        filterAndDisplayPredictions();
        
    } catch (error) {
        console.error('加载预测结果失败:', error);
        document.getElementById('predictionsTable').innerHTML = 
            `<tr><td colspan="6" class="text-center text-danger">加载失败: ${error.message}</td></tr>`;
    }
}

// 过滤并显示预测结果
function filterAndDisplayPredictions() {
    // 应用过滤条件
    const filteredPredictions = allPredictions.filter(p => p.predicted_prob >= currentFilter);
    
    // 分页显示
    displayPredictions(filteredPredictions, currentPage);
}

// 显示预测结果
function displayPredictions(predictions, page) {
    const tableBody = document.getElementById('predictionsTable');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    // 计算分页
    const totalPages = Math.ceil(predictions.length / pageSize);
    const start = (page - 1) * pageSize;
    const end = Math.min(start + pageSize, predictions.length);
    const pagePredictions = predictions.slice(start, end);
    
    // 生成表格行
    if (pagePredictions.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" class="text-center">没有符合条件的预测结果</td></tr>';
    } else {
        for (const prediction of pagePredictions) {
            const row = document.createElement('tr');
            
            // SKU ID
            const skuIdCell = document.createElement('td');
            skuIdCell.textContent = prediction.sku_id;
            row.appendChild(skuIdCell);
            
            // 商品名称
            const nameCell = document.createElement('td');
            nameCell.textContent = prediction.sku_name || `商品_${prediction.sku_id}`;
            row.appendChild(nameCell);
            
            // 当前价格
            const priceCell = document.createElement('td');
            priceCell.textContent = `¥${prediction.current_price.toFixed(2)}`;
            row.appendChild(priceCell);
            
            // 变动概率
            const probCell = document.createElement('td');
            const probValue = (prediction.predicted_prob * 100).toFixed(2) + '%';
            probCell.textContent = probValue;
            
            // 根据概率设置颜色
            if (prediction.predicted_prob >= 0.8) {
                probCell.className = 'text-danger fw-bold';
            } else if (prediction.predicted_prob >= 0.5) {
                probCell.className = 'text-warning fw-bold';
            } else {
                probCell.className = 'text-success';
            }
            
            row.appendChild(probCell);
            
            // 最后更新时间
            const updateCell = document.createElement('td');
            updateCell.textContent = formatDateTime(prediction.last_updated);
            row.appendChild(updateCell);
            
            // 操作按钮
            const actionCell = document.createElement('td');
            const viewBtn = document.createElement('button');
            viewBtn.className = 'btn btn-sm btn-outline-primary';
            viewBtn.innerHTML = '<i class="bi bi-eye"></i>';
            viewBtn.addEventListener('click', () => showSkuDetail(prediction));
            actionCell.appendChild(viewBtn);
            row.appendChild(actionCell);
            
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
    
    // 如果只有一页，不显示分页控件
    if (totalPages <= 1) return;
    
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
            currentPage--;
            filterAndDisplayPredictions();
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
            currentPage = i;
            filterAndDisplayPredictions();
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
            currentPage++;
            filterAndDisplayPredictions();
        }
    });
    nextItem.appendChild(nextLink);
    pagination.appendChild(nextItem);
}

// 加载预测统计
async function loadPredictionStats() {
    try {
        const response = await fetch('/api/predictions/stats', {
            headers: getAuthHeaders()
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                showLoginModal();
                return;
            }
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const stats = await response.json();
        displayPredictionStats(stats);
        
    } catch (error) {
        console.error('加载预测统计失败:', error);
    }
}

// 显示预测统计
function displayPredictionStats(stats) {
    // 更新统计数字
    document.getElementById('totalPredictions').textContent = stats.total_predictions || 0;
    document.getElementById('highRiskCount').textContent = stats.high_prob_count || 0;
    
    // 绘制时间分布图表
    drawTimeDistributionChart(stats.time_distribution);
}

// 绘制时间分布图表
function drawTimeDistributionChart(timeDistribution) {
    const ctx = document.getElementById('timeDistChart');
    if (!ctx) return;
    
    // 销毁旧图表
    if (timeDistChart) {
        timeDistChart.destroy();
    }
    
    // 准备数据
    const hours = timeDistribution.map(item => item.hour);
    const counts = timeDistribution.map(item => item.count);
    
    // 创建新图表
    timeDistChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hours,
            datasets: [{
                label: '预测数量',
                data: counts,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '预测时间分布'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '数量'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '小时'
                    }
                }
            }
        }
    });
}

// 显示SKU详情
async function showSkuDetail(prediction) {
    try {
        // 获取SKU详细信息
        const response = await fetch(`/api/predictions/sku/${prediction.sku_id}`, {
            headers: getAuthHeaders()
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const skuDetail = await response.json();
        
        // 更新模态框内容
        document.getElementById('modalSkuId').textContent = skuDetail.sku_id;
        document.getElementById('modalSkuName').textContent = skuDetail.current_prediction.sku_name || `商品_${skuDetail.sku_id}`;
        const price = Number(skuDetail.current_prediction.current_price);
        document.getElementById('modalCurrentPrice').textContent = `¥${isNaN(price) ? '0.00' : price.toFixed(2)}`;
        
        const probElement = document.getElementById('modalPredictedProb');
        const probValue = (skuDetail.current_prediction.predicted_prob * 100).toFixed(2) + '%';
        probElement.textContent = probValue;
        
        // 根据概率设置颜色
        if (skuDetail.current_prediction.predicted_prob >= 0.8) {
            probElement.className = 'text-danger fw-bold';
        } else if (skuDetail.current_prediction.predicted_prob >= 0.5) {
            probElement.className = 'text-warning fw-bold';
        } else {
            probElement.className = 'text-success';
        }
        
        // 绘制价格历史图表
        drawPriceHistoryChart(skuDetail.price_history);
        
        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('skuDetailModal'));
        modal.show();
        
    } catch (error) {
        console.error('获取SKU详情失败:', error);
        alert('获取SKU详情失败: ' + error.message);
    }
}

// 绘制价格历史图表
function drawPriceHistoryChart(priceHistory) {
    const ctx = document.getElementById('modalPriceChart');
    if (!ctx) return;
    
    // 销毁旧图表
    if (skuDetailChart) {
        skuDetailChart.destroy();
    }
    
    // 准备数据
    const dates = priceHistory.map(item => formatDate(item.date));
    const prices = priceHistory.map(item => item.price);
    const promotions = priceHistory.map(item => item.is_promotion ? 'red' : 'blue');
    
    // 创建新图表
    skuDetailChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: '价格',
                data: prices,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                pointBackgroundColor: promotions,
                pointRadius: 4,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '价格历史'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const index = context.dataIndex;
                            const isPromotion = priceHistory[index].is_promotion;
                            return `价格: ¥${context.raw.toFixed(2)}${isPromotion ? ' (促销)' : ''}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: '价格 (¥)'
                    }
                }
            }
        }
    });
}

// 格式化日期时间
function formatDateTime(dateTimeStr) {
    const date = new Date(dateTimeStr);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// 格式化日期
function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('zh-CN', {
        month: '2-digit',
        day: '2-digit'
    });
}

// 实时预测功能
async function predictRealtime() {
    const skuInput = document.getElementById('skuInput').value.trim();
    const skuList = skuInput.split(',').map(s => s.trim()).filter(Boolean);
    const date = document.getElementById('endDateInput').value || (new Date()).toISOString().slice(0,10);

    // 预测前清空结果区
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';

    if (skuList.length === 0) {
        alert('请输入至少一个SKU');
        return;
    }

    const btn = document.getElementById('realtimePredictBtn');
    btn.disabled = true;
    btn.textContent = '预测中...';

    try {
        const response = await fetch('/api/predictions/realtime', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sku_list: skuList,
                date: date
            })
        });
        if (!response.ok) throw new Error('预测请求失败');
        const results = await response.json();
        displayPredictionResults(results);
    } catch (e) {
        alert('预测失败: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '实时预测';
    }
}

function displayPredictionResults(results) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';
    if (!results || results.length === 0) {
        container.innerHTML = '<div class="alert alert-warning">无预测结果</div>';
        return;
    }
    results.forEach(res => {
        const color = res.predict_proba === -1 ? 'text-danger' : (res.predict_proba >= 0.8 ? 'text-danger' : (res.predict_proba >= 0.5 ? 'text-warning' : 'text-success'));
        container.innerHTML += `
            <div class="card mb-2">
                <div class="card-body">
                    <b>SKU:</b> ${res.sku} <b>日期:</b> ${res.date}
                    <span class="${color} ms-3"><b>${res.sampling_plan}</b></span>
                    ${res.predict_proba !== -1 ? `<span class="ms-3">概率: ${(res.predict_proba*100).toFixed(2)}%</span>` : ''}
                </div>
            </div>
        `;
    });
}