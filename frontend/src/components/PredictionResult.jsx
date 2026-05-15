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

  // In this dataset, 0 = At Risk/High Risk, 1 = Healthy/Low Risk
  const models = [result.knn, result.logistic_regression, result.svm, result.random_forest].filter(Boolean);
  const totalModels = models.length;
  const highRiskCount = models.filter(m => m.prediction === 0).length;
  const healthyCount = models.filter(m => m.prediction === 1).length;
  
  // Final decision logic
  let statusText = "";
  let statusClass = "";
  let isHighRisk = false;
  let isConflicting = false;
  let isGuidelineOverride = false;

  // Clinical Red Flags based on the user's expert input
  const hasCriticalMarkers = result.input_data && (
    result.input_data.ca > 0 || 
    result.input_data.thal <= 2
  );

  if (highRiskCount === totalModels || (highRiskCount > 0 && hasCriticalMarkers)) {
    statusText = "Nguy Cơ Cao";
    statusClass = "status-high";
    isHighRisk = true;
    isConflicting = highRiskCount !== totalModels;
  } else if (healthyCount === totalModels) {
    statusText = "Bình Thường";
    statusClass = "status-low";
    isHighRisk = false;
  } else {
    // Conflict between models - Potential Risk
    statusText = "Nguy Cơ Tiềm Ẩn";
    statusClass = "status-warning";
    isHighRisk = true; 
    isConflicting = true;
  }


  // =============================================
  // GUIDELINE 2.5 INTEGRATION (from Backend)
  // =============================================
  const guidelineRisk = result.guideline_2_5 ? {
    ...result.guideline_2_5,
    className: ["status-low", "status-warning", "status-high", "status-danger"][result.guideline_2_5.level]
  } : null;

  // Check if AI models' status texts agree with Guideline
  const modelStatuses = models.map(m => m.status || "");
  const modelsAgreeWithGuideline = guidelineRisk && modelStatuses.every(s => {
    if (guidelineRisk.level === 0) return s.includes("Bình thường");
    if (guidelineRisk.level === 1) return s.includes("trung bình");
    if (guidelineRisk.level === 2) return s.includes("cao") && !s.includes("rất cao");
    if (guidelineRisk.level === 3) return s.includes("rất cao");
    return false;
  });

  // Override status based on Guideline if needed
  if (guidelineRisk && guidelineRisk.level >= 1) {
    if (!isHighRisk || guidelineRisk.level >= 2) {
      statusText = guidelineRisk.label.toUpperCase();
      statusClass = guidelineRisk.className;
      isHighRisk = guidelineRisk.level >= 2;
    }
  }

  // =============================================
  // CONFIDENCE CALCULATION
  // =============================================
  const avgRiskProb = models.reduce((acc, m) => {
    const p = m.prediction === 0 ? (m.probability || 0.8) : (1 - (m.probability || 0.8));
    return acc + p;
  }, 0) / totalModels;

  let finalConfidence = isHighRisk ? avgRiskProb : (1 - avgRiskProb);
  let confidenceMessage = "Độ tin cậy tổng hợp";

  if (modelsAgreeWithGuideline) {
    // AI and Guideline AGREE → High confidence!
    finalConfidence = Math.max(finalConfidence, 0.85);
    confidenceMessage = "Độ tin cậy tổng hợp";
    isConflicting = false;
    isGuidelineOverride = false;
  } else if (guidelineRisk && guidelineRisk.level >= 1 && healthyCount === totalModels) {
    // AI says healthy (prediction=1) but Guideline says risk → Override with explanation
    isGuidelineOverride = true;
    finalConfidence = 0.75 + (finalConfidence * 0.2);
    confidenceMessage = "Độ tin cậy tổng hợp (Phác đồ ghi đè)";
  } else if (isConflicting) {
    // Models genuinely disagree with each other
    finalConfidence = Math.min(finalConfidence, 0.75);
    confidenceMessage = "⚠️ Độ tin cậy thấp do các mô hình không đồng nhất";
  }

  let displayedProbability = Math.round(finalConfidence * 1000) / 10;

  const hasCriticalSymptoms = result.input_data && (
    result.input_data.cp === 0 || 
    (result.input_data.age > 60 && result.input_data.oldpeak > 1.0) ||
    result.input_data.exang === 1 ||
    result.input_data.thal <= 2
  );

  if (hasCriticalSymptoms && !isHighRisk && !modelsAgreeWithGuideline) {
    displayedProbability = Math.min(displayedProbability, 45);
    confidenceMessage = "⚠️ Độ tin cậy thấp do mâu thuẫn lâm sàng";
  }

  return (
    <div className="result-card glass-panel animate-fade-in">
      <h3 className="result-title">Kết Quả Đánh Giá</h3>
      
      <div className="result-content">
        <div className={`status-badge ${statusClass}`}>
          {statusText}
        </div>
        
        <p style={{color: 'var(--text-secondary)', marginBottom: '0.5rem'}}>
          {isGuidelineOverride 
            ? "⚠️ Mô hình báo bình thường nhưng Phác đồ Y khoa (2.5) phát hiện nguy cơ tiềm ẩn."
            : (isConflicting 
                ? "Các mô hình chẩn đoán đang có sự bất đồng. Hệ thống đã đối chiếu thêm với Phác đồ 2.5."
                : `Dựa trên dữ liệu và Phác đồ 2.5, hệ thống dự đoán bạn ${isHighRisk ? 'có dấu hiệu' : 'không có dấu hiệu'} của bệnh tim mạch.`
              )
          }
        </p>
        {guidelineRisk && (
          <p style={{fontSize: '0.8rem', color: 'var(--primary)', fontWeight: 'bold', marginBottom: '1rem'}}>
            Phân tầng theo Phác đồ 2.5: {guidelineRisk.label}
          </p>
        )}

        <div className="risk-meter">
          <div 
            className={`risk-fill ${isConflicting ? 'risk-warning' : (isHighRisk ? 'risk-high' : 'risk-low')}`} 
            style={{width: `${displayedProbability}%`}}
          ></div>
        </div>
        <p style={{
          textAlign: 'right', 
          fontSize: '0.875rem', 
          color: hasCriticalSymptoms && !isHighRisk ? 'var(--danger)' : 'var(--text-secondary)',
          fontWeight: hasCriticalSymptoms && !isHighRisk ? 'bold' : 'normal'
        }}>
          {confidenceMessage}: {displayedProbability}%
        </p>

        <div className="recommendation-box" style={{
          marginTop: '1.5rem', 
          padding: '1rem', 
          backgroundColor: isHighRisk || hasCriticalSymptoms ? 'rgba(255, 71, 87, 0.1)' : 'rgba(255,255,255,0.05)', 
          borderRadius: '8px',
          borderLeft: `4px solid ${isHighRisk || hasCriticalSymptoms ? 'var(--danger)' : 'var(--primary)'}`
        }}>
          <h4 style={{fontSize: '0.9rem', marginBottom: '0.5rem', color: isHighRisk || hasCriticalSymptoms ? 'var(--danger)' : 'var(--primary)'}}>
            {hasCriticalSymptoms && !isHighRisk ? '⚠️ CẢNH BÁO LÂM SÀNG QUAN TRỌNG:' : 'Khuyến nghị lâm sàng:'}
          </h4>
          <p style={{fontSize: '0.85rem', fontWeight: (hasCriticalSymptoms || isHighRisk) ? '600' : '400'}}>
            {hasCriticalSymptoms && healthyCount === totalModels 
              ? "Mặc dù mô hình báo nguy cơ thấp, nhưng triệu chứng lâm sàng của bạn là rất đáng ngại. Bạn cần đi khám bác sĩ ngay lập tức để làm điện tâm đồ và xét nghiệm men tim."
              : (highRiskCount === totalModels
                  ? "CẢNH BÁO KHẨN CẤP: Các chỉ số của bạn đang ở mức nguy hiểm cao. Hãy đến cơ sở y tế gần nhất ngay lập tức để được thăm khám."
                  : (isConflicting
                      ? "Có sự bất đồng giữa các mô hình chẩn đoán. Chúng tôi khuyên bạn nên thực hiện thêm các xét nghiệm chuyên sâu như chụp CT mạch vành hoặc siêu âm tim."
                      : "Tuyệt vời! Các chỉ số của bạn đều nằm trong ngưỡng an toàn. Hãy tiếp tục duy trì chế độ ăn uống và tập luyện hiện tại."
                    )
                )
            }
          </p>
        </div>

        <div className="model-details">
          <h4 style={{marginBottom: '0.5rem', fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)'}}>
            Chi tiết mô hình & Phân tầng
          </h4>
          {[
            { name: "K-Nearest Neighbors (KNN)", data: result.knn },
            { name: "Logistic Regression", data: result.logistic_regression },
            { name: "SVM", data: result.svm },
            { name: "Random Forest (RF)", data: result.random_forest }
          ].filter(item => item.data).map((m, idx) => {
            const status = m.data.status || "";
            let color = 'var(--success)';
            if (status.includes("rất cao")) color = 'var(--danger)';
            else if (status.includes("cao")) color = 'var(--danger)';
            else if (status.includes("trung bình")) color = 'var(--warning)';
            
            return (
              <div key={idx} className="model-item">
                <span>{m.name}:</span>
                <span style={{color, fontWeight: '600'}}>
                  {status}
                </span>
              </div>
            );
          })}
        </div>
        
        <p style={{fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '1.5rem', fontStyle: 'italic'}}>
          * Lưu ý: Kết quả này chỉ mang tính chất tham khảo dựa trên mô hình học máy. Hãy luôn tham khảo ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác nhất.
        </p>
      </div>
    </div>
  );
}
