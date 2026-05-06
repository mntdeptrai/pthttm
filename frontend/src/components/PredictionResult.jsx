import React from 'react';

export default function PredictionResult({ result, loading, error }) {
  if (loading) {
    return (
      <div className="result-card glass-panel animate-fade-in">
        <div className="result-placeholder">
          <svg className="animate-spin" width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 4V2M12 22V20M4 12H2M22 12H20M5.63604 5.63604L4.22183 4.22183M19.7782 19.7782L18.364 18.364M5.63604 18.364L4.22183 19.7782M19.7782 4.22183L18.364 5.63604" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <p>Đang phân tích dữ liệu...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="result-card glass-panel animate-fade-in">
        <h3 className="result-title" style={{color: 'var(--danger)'}}>Lỗi Phân Tích</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-card glass-panel">
        <div className="result-placeholder">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m3.75 9v6m3-3H9m1.5-12H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
          </svg>
          <p>Nhập thông tin và nhấn "Dự Đoán" để xem kết quả</p>
        </div>
      </div>
    );
  }

  // We primarily use KNN as it was indicated to have highest accuracy
  const primaryResult = result.knn;
  const isHighRisk = primaryResult.prediction === 1;
  const probability = primaryResult.probability ? Math.round(primaryResult.probability * 100) : (isHighRisk ? 85 : 15);

  return (
    <div className="result-card glass-panel animate-fade-in">
      <h3 className="result-title">Kết Quả Đánh Giá</h3>
      
      <div className="result-content">
        <div className={`status-badge ${isHighRisk ? 'status-high' : 'status-low'}`}>
          {isHighRisk ? 'Nguy Cơ Cao' : 'Nguy Cơ Thấp'}
        </div>
        
        <p style={{color: 'var(--text-secondary)'}}>
          Dựa trên dữ liệu, hệ thống dự đoán bạn {isHighRisk ? 'có dấu hiệu' : 'không có dấu hiệu'} của bệnh tăng huyết áp.
        </p>

        <div className="risk-meter">
          <div 
            className={`risk-fill ${isHighRisk ? 'risk-high' : 'risk-low'}`} 
            style={{width: `${probability}%`}}
          ></div>
        </div>
        <p style={{textAlign: 'right', fontSize: '0.875rem', color: 'var(--text-secondary)'}}>
          Độ tin cậy: {probability}%
        </p>

        <div className="model-details">
          <h4 style={{marginBottom: '0.5rem', fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)'}}>
            Chi tiết mô hình
          </h4>
          <div className="model-item">
            <span>K-Nearest Neighbors (KNN):</span>
            <span style={{color: result.knn.prediction === 1 ? 'var(--danger)' : 'var(--success)'}}>
              {result.knn.status}
            </span>
          </div>
          <div className="model-item">
            <span>Logistic Regression:</span>
            <span style={{color: result.logistic_regression.prediction === 1 ? 'var(--danger)' : 'var(--success)'}}>
              {result.logistic_regression.status}
            </span>
          </div>
        </div>
        
        <p style={{fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '1.5rem', fontStyle: 'italic'}}>
          * Lưu ý: Kết quả này chỉ mang tính chất tham khảo dựa trên mô hình học máy. Hãy luôn tham khảo ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác nhất.
        </p>
      </div>
    </div>
  );
}
