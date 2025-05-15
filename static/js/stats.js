// 全局变量
let samplingChart = null;
let crawlChart = null;
let daysFilter = 7; // 默认显示最近7天数据

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 检查登录状态
    checkAuthStatus();
    
    // 加载统计信息
    loadStats();
    
    // 绑定刷新按钮事件
    document.getElementById('refreshBtn')?.addEventListener('click', loadStats);
    
    // 绑定时间范围下拉菜单事件
    const dropdownItems = document.querySelectorAll('.dropdown-item[data-days]');
    dropdownItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            daysFilter = parseInt(this.getAttribute('data-days'));
            loadStats();
        });
    });
});

// 加载统计信息
async function loadStats() {
    try {
        // 计算日期范围
        const endDate = new Date();
        const startDate = new Date();
        startDate.setDate(endDate.getDate() - (daysFilter > 0 ? daysFilter : 365 * 5)); // 如果daysFilter为0，显示全部数据
        
        // 获取采样计划统计
        const samplingResponse = await fetch(`/api/sampling-plans/stats?start_date=${startDate.toISOString()}&end_date=${endDate.toISOString()}`, {
            headers: getAuthHeaders()
        });
        
        if (!samplingResponse.ok) {
            if (samplingResponse.status === 401) {
                showLoginModal();
                return;
            }
            throw new Error(`获取采样统计失败: ${samplingResponse.status}`);
        }
        
        const samplingStats = await samplingResponse.json();
        
        // 获取爬取统计
        const crawlResponse = await fetch(`/api/crawl-results/stats?start_date=${startDate.toISOString()}&end_date=${endDate.toISOString()}`, {
            headers: getAuthHeaders()
        });
        
        if (!crawlResponse.ok) {
            if (crawlResponse.status === 401) {
                showLoginModal();
                return;
            }
            throw new Error(`获取爬取统计失败: ${crawlResponse.status}`);
        }
        
        const crawlStats = await crawlResponse.json();
        
        // 显示统计信息
        displayStats(samplingStats, crawlStats);
        
        // 绘制图表
        drawCharts(samplingStats, crawlStats);
        
    } catch (error) {
        console.error('加载统计信息失败:', error);
        alert('加载统计信息失败: ' + error.message);
    }
}

// 显示统计信息
function displayStats(samplingStats, crawlStats) {
    // 采样计划统计
    const avgSamplingRate = document.getElementById('avgSamplingRate');
    const avgCostSaving = document.getElementById('avgCostSaving');
    
    if (avgSamplingRate) {
        const avgRate = samplingStats.strata_stats?.reduce((sum, stat) => sum + stat.sampling_rate, 0) / samplingStats.strata_stats?.length || 0;
        avgSamplingRate.textContent = (avgRate * 100).toFixed(2) + '%';
    }
    
    if (avgCostSaving) {
        avgCostSaving.textContent = (samplingStats.cost_saving * 100).toFixed(2) + '%';
    }
    
    // 爬取统计
    const totalCrawls = document.getElementById('totalCrawls');
    const uniqueSkus = document.getElementById('uniqueSkus');
    
    if (totalCrawls) totalCrawls.textContent = crawlStats.total_crawls || 0;
    if (uniqueSkus) uniqueSkus.textContent = crawlStats.unique_skus || 0;
    
    // 分层采样统计
    const strataTable = document.getElementById('strataTable');
    if (strataTable && samplingStats.strata_stats) {
        strataTable.innerHTML = '';
        
        samplingStats.strata_stats.forEach(stat => {
            const row = document.createElement('tr');
            
            // 类别
            const categoryCell = document.createElement('td');
            categoryCell.textContent = `类别 ${stat.cluster_id + 1}`;
            row.appendChild(categoryCell);
            
            // 总SKU数
            const totalCell = document.createElement('td');
            totalCell.textContent = stat.total;
            row.appendChild(totalCell);
            
            // 采样SKU数
            const sampledCell = document.createElement('td');
            sampledCell.textContent = stat.sampled;
            row.appendChild(sampledCell);
            
            // 采样率
            const rateCell = document.createElement('td');
            rateCell.textContent = (stat.sampling_rate * 100).toFixed(2) + '%';
            row.appendChild(rateCell);
            
            strataTable.appendChild(row);
        });
    }
}

// 绘制图表
function drawCharts(samplingStats, crawlStats) {
    const samplingCtx = document.getElementById('samplingChart');
    const crawlCtx = document.getElementById('crawlChart');
    
    // 销毁旧图表
    if (samplingChart) samplingChart.destroy();
    if (crawlChart) crawlChart.destroy();
    
    // 采样计划图表
    if (samplingCtx && samplingStats.strata_stats) {
        samplingChart = new Chart(samplingCtx, {
            type: 'bar',
            data: {
                labels: samplingStats.strata_stats.map(stat => `类别 ${stat.cluster_id + 1}`),
                datasets: [{
                    label: '采样率',
                    data: samplingStats.strata_stats.map(stat => stat.sampling_rate * 100),
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: '采样率 (%)'
                        }
                    }
                }
            }
        });
    }
    
    // 爬取统计图表
    if (crawlCtx && crawlStats.daily_stats) {
        // 按日期排序
        const sortedDailyStats = [...crawlStats.daily_stats].sort((a, b) => new Date(a.date) - new Date(b.date));
        
        crawlChart = new Chart(crawlCtx, {
            type: 'line',
            data: {
                labels: sortedDailyStats.map(stat => formatDate(stat.date)),
                datasets: [
                    {
                        label: '每日爬取次数',
                        data: sortedDailyStats.map(stat => stat.crawl_count),
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                        tension: 0.1
                    },
                    {
                        label: '每日唯一SKU数',
                        data: sortedDailyStats.map(stat => stat.unique_skus),
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '数量'
                        }
                    }
                }
            }
        });
    }
}