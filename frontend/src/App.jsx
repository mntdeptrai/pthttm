import { useState } from 'react'
import './App.css'
import PredictionResult from './components/PredictionResult'

function App() {
  const [formData, setFormData] = useState({
    age: '', sex: '', cp: '', trestbps: '', chol: '', 
    fbs: '', restecg: '', thalach: '', exang: '', 
    oldpeak: '', slope: '', ca: '', thal: ''
  });
  
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    // Convert string values to numbers
    const numericData = {};
    for (const key in formData) {
      numericData[key] = parseFloat(formData[key]);
      if (isNaN(numericData[key])) {
        setError(`Vui lòng nhập số hợp lệ cho trường: ${key}`);
        setLoading(false);
        return;
      }
    }

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(numericData),
      });

      if (!response.ok) {
        throw new Error('Không thể kết nối đến máy chủ dự đoán');
      }

      const data = await response.json();
      // Simulate slight delay for better UX animation
      setTimeout(() => {
        setResult(data);
        setLoading(false);
      }, 600);
      
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const formFields = [
    { name: 'age', label: 'Tuổi', type: 'number', placeholder: 'VD: 45' },
    { name: 'sex', label: 'Giới tính (1: Nam, 0: Nữ)', type: 'number', placeholder: '0 hoặc 1', min: "0", max: "1" },
    { name: 'cp', label: 'Loại đau ngực (0-3)', type: 'number', placeholder: '0, 1, 2, hoặc 3', min: "0", max: "3" },
    { name: 'trestbps', label: 'Huyết áp lúc nghỉ (mmHg)', type: 'number', placeholder: 'VD: 120' },
    { name: 'chol', label: 'Cholesterol (mg/dl)', type: 'number', placeholder: 'VD: 200' },
    { name: 'fbs', label: 'Đường huyết đói > 120 mg/dl (1: Đúng, 0: Sai)', type: 'number', placeholder: '0 hoặc 1', min: "0", max: "1" },
    { name: 'restecg', label: 'Điện tâm đồ lúc nghỉ (0-2)', type: 'number', placeholder: '0, 1, hoặc 2', min: "0", max: "2" },
    { name: 'thalach', label: 'Nhịp tim tối đa', type: 'number', placeholder: 'VD: 150' },
    { name: 'exang', label: 'Đau thắt ngực khi vận động (1: Có, 0: Không)', type: 'number', placeholder: '0 hoặc 1', min: "0", max: "1" },
    { name: 'oldpeak', label: 'Độ chênh ST', type: 'number', step: "0.1", placeholder: 'VD: 1.5' },
    { name: 'slope', label: 'Độ dốc đoạn ST (0-2)', type: 'number', placeholder: '0, 1, hoặc 2', min: "0", max: "2" },
    { name: 'ca', label: 'Số mạch máu chính (0-4)', type: 'number', placeholder: '0, 1, 2, 3, hoặc 4', min: "0", max: "4" },
    { name: 'thal', label: 'Kết quả Thallium (1-3)', type: 'number', placeholder: '1, 2, hoặc 3', min: "1", max: "3" }
  ];

  return (
    <div className="app-container">
      <header className="header animate-fade-in">
        <h1 className="text-gradient">Chẩn Đoán Tăng Huyết Áp AI</h1>
        <p>Hệ thống sử dụng mô hình học máy để hỗ trợ đánh giá nguy cơ mắc bệnh dựa trên 13 chỉ số y tế lâm sàng.</p>
      </header>

      <main className="main-content">
        <div className="form-card glass-panel animate-fade-in" style={{animationDelay: '0.1s'}}>
          <h2 style={{marginBottom: '1.5rem', fontSize: '1.25rem'}}>Thông tin Bệnh nhân</h2>
          <form onSubmit={handleSubmit} className="form-grid">
            {formFields.map((field) => (
              <div key={field.name} className="input-group">
                <label className="input-label" htmlFor={field.name}>
                  {field.label}
                </label>
                <input
                  id={field.name}
                  name={field.name}
                  type={field.type}
                  step={field.step}
                  min={field.min}
                  max={field.max}
                  placeholder={field.placeholder}
                  value={formData[field.name]}
                  onChange={handleInputChange}
                  className="input-field"
                  required
                />
              </div>
            ))}
            <div style={{gridColumn: '1 / -1', marginTop: '1rem'}}>
              <button 
                type="submit" 
                className="btn btn-primary" 
                style={{width: '100%'}}
                disabled={loading}
              >
                {loading ? 'Đang Phân Tích...' : 'Dự Đoán Nguy Cơ'}
              </button>
            </div>
          </form>
        </div>

        <div className="result-section">
          <PredictionResult result={result} loading={loading} error={error} />
        </div>
      </main>
    </div>
  )
}

export default App
