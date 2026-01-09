import React, { useState } from 'react'

export default function HeartDiseaseAssessment(){
  const [activeTab, setActiveTab] = useState('demographics')
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)
  const [formData, setFormData] = useState({
    age:55, sex:'male', bmi:27,
    systolic_bp:130, diastolic_bp:80, heart_rate:75, prevalent_hypertension:0,
    total_cholesterol:200, hdl:50, ldl:120, triglycerides:150, fasting_glucose:95, diabetes:0,
    sodium:140, potassium:4.2, calcium:9.5, creatinine:1.0, egfr:90,
    smoking:0, physical_activity:'moderate', family_history:0
  })

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target
    setFormData(prev=>({...prev, [name]: type==='checkbox' ? (checked?1:0) : (type==='number'? Number(value): value)}))
  }

  const handleSubmit = async () => {
    setLoading(true); setError(null); setPrediction(null)
    // Use environment variable for API URL, or production Render URL
    const backendBase = import.meta.env.VITE_API_URL || 'https://heart-disease-platform.onrender.com';

    try{
      console.log('Calling API:', `${backendBase}/api/predict`);
      const res = await fetch(`${backendBase}/api/predict`, {
        method:'POST', 
        headers:{'Content-Type':'application/json'}, 
        body: JSON.stringify(formData)
      })
      console.log('Response status:', res.status);
      
      const responseText = await res.text();
      console.log('Response text:', responseText);
      
      if(!res.ok){
        let errText = 'Prediction failed';
        try{ 
          const errJson = JSON.parse(responseText); 
          errText = errJson.detail || errText 
        } catch(e){ 
          errText = responseText || errText 
        }
        throw new Error(errText)
      }
      
      const json = JSON.parse(responseText);
      setPrediction(json)
    }catch(err){ 
      console.error('Error:', err);
      setError(err.message) 
    }
    finally{ setLoading(false) }
  }

  const tabs = [
    {id:'demographics', label:'Demographics'},
    {id:'cardiovascular', label:'Cardiovascular'},
    {id:'metabolic', label:'Metabolic'},
    {id:'labs', label:'Lab Results'}
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-6">Heart Disease Risk Assessment</h1>
        
        {/* Tabs */}
        <div className="bg-white rounded-lg shadow-md mb-6">
          <div className="flex border-b">
            {tabs.map(tab=>(
              <button key={tab.id} onClick={()=>setActiveTab(tab.id)} 
                className={`px-6 py-3 font-medium ${activeTab===tab.id?'border-b-2 border-blue-500 text-blue-600':'text-gray-600'}`}>
                {tab.label}
              </button>
            ))}
          </div>

          <div className="p-6">
            {activeTab==='demographics'&&(
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div><label className="block mb-1 font-medium">Age</label>
                  <input type="number" name="age" value={formData.age} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">Sex</label>
                  <select name="sex" value={formData.sex} onChange={handleInputChange} className="w-full border rounded px-3 py-2">
                    <option value="male">Male</option><option value="female">Female</option>
                  </select></div>
                <div><label className="block mb-1 font-medium">BMI</label>
                  <input type="number" step="0.1" name="bmi" value={formData.bmi} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">Physical Activity</label>
                  <select name="physical_activity" value={formData.physical_activity} onChange={handleInputChange} className="w-full border rounded px-3 py-2">
                    <option value="sedentary">Sedentary</option><option value="light">Light</option>
                    <option value="moderate">Moderate</option><option value="active">Active</option>
                    <option value="very_active">Very Active</option>
                  </select></div>
                <div><label className="flex items-center"><input type="checkbox" name="smoking" checked={formData.smoking===1} onChange={handleInputChange} className="mr-2"/>Smoking</label></div>
                <div><label className="flex items-center"><input type="checkbox" name="family_history" checked={formData.family_history===1} onChange={handleInputChange} className="mr-2"/>Family History</label></div>
              </div>
            )}

            {activeTab==='cardiovascular'&&(
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div><label className="block mb-1 font-medium">Systolic BP</label>
                  <input type="number" name="systolic_bp" value={formData.systolic_bp} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">Diastolic BP</label>
                  <input type="number" name="diastolic_bp" value={formData.diastolic_bp} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">Heart Rate</label>
                  <input type="number" name="heart_rate" value={formData.heart_rate} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="flex items-center"><input type="checkbox" name="prevalent_hypertension" checked={formData.prevalent_hypertension===1} onChange={handleInputChange} className="mr-2"/>Hypertension</label></div>
              </div>
            )}

            {activeTab==='metabolic'&&(
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div><label className="block mb-1 font-medium">Total Cholesterol</label>
                  <input type="number" name="total_cholesterol" value={formData.total_cholesterol} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">HDL</label>
                  <input type="number" name="hdl" value={formData.hdl} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">LDL</label>
                  <input type="number" name="ldl" value={formData.ldl} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">Triglycerides</label>
                  <input type="number" name="triglycerides" value={formData.triglycerides} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">Fasting Glucose</label>
                  <input type="number" name="fasting_glucose" value={formData.fasting_glucose} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="flex items-center"><input type="checkbox" name="diabetes" checked={formData.diabetes===1} onChange={handleInputChange} className="mr-2"/>Diabetes</label></div>
              </div>
            )}

            {activeTab==='labs'&&(
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div><label className="block mb-1 font-medium">Sodium</label>
                  <input type="number" step="0.1" name="sodium" value={formData.sodium} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">Potassium</label>
                  <input type="number" step="0.1" name="potassium" value={formData.potassium} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">Calcium</label>
                  <input type="number" step="0.1" name="calcium" value={formData.calcium} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">Creatinine</label>
                  <input type="number" step="0.1" name="creatinine" value={formData.creatinine} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
                <div><label className="block mb-1 font-medium">eGFR</label>
                  <input type="number" name="egfr" value={formData.egfr} onChange={handleInputChange} className="w-full border rounded px-3 py-2"/></div>
              </div>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div className="text-center mb-6">
          <button onClick={handleSubmit} disabled={loading} 
            className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50">
            {loading?'Analyzing...':'Assess Risk'}
          </button>
        </div>

        {/* Error */}
        {error&&<div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">{error}</div>}

        {/* Results */}
        {prediction&&(
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-bold mb-4">Risk Assessment Results</h2>
            
            {/* Risk Level with Category */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <p className="text-lg">Overall Risk Probability:</p>
                <span className="font-bold text-3xl">{(prediction.probability*100).toFixed(1)}%</span>
              </div>
              {prediction.risk_category&&(
                <div className={`inline-block px-4 py-2 rounded-lg font-semibold ${
                  prediction.risk_category==='High'?'bg-red-100 text-red-800':
                  prediction.risk_category==='Medium'?'bg-yellow-100 text-yellow-800':
                  'bg-green-100 text-green-800'
                }`}>
                  Risk Level: {prediction.risk_category}
                </div>
              )}
            </div>
            
            {/* Modality Contributions */}
            {prediction.modalities&&Object.keys(prediction.modalities).length>0&&(
              <div className="mb-6 pb-6 border-b">
                <h3 className="font-semibold text-lg mb-3">Modality Contributions:</h3>
                <div className="space-y-2">
                  {Object.entries(prediction.modalities).map(([key,val])=>(
                    <div key={key} className="flex items-center">
                      <span className="inline-block w-40 capitalize">{key}:</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-4 mr-3">
                        <div className="bg-blue-600 h-4 rounded-full" style={{width:`${(val*100).toFixed(1)}%`}}></div>
                      </div>
                      <span className="font-medium w-16 text-right">{(val*100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Feature Importance */}
            {prediction.feature_importance&&prediction.feature_importance.length>0&&(
              <div className="mb-6 pb-6 border-b">
                <h3 className="font-semibold text-lg mb-3">Key Risk Factors (Deviation from Normal):</h3>
                <div className="space-y-1">
                  {prediction.feature_importance.slice(0,6).map((f,i)=>(
                    <div key={i} className="flex justify-between">
                      <span className="text-gray-700">{f.name.replace(/_/g,' ')}:</span>
                      <span className="font-medium">{f.value.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {prediction.recommendations&&prediction.recommendations.length>0&&(
              <div>
                <h3 className="font-semibold text-lg mb-3">Personalized Recommendations:</h3>
                <ul className="list-disc list-inside space-y-2">
                  {prediction.recommendations.map((rec,i)=>(
                    <li key={i} className="text-gray-700">{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
