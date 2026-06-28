import React, { useState } from 'react';

export default function DemoDatasetSelector({ onSelectDataset, loading }) {
  const [selected, setSelected] = useState('');
  
  const datasets = [
    { id: 'm5', name: 'M5 Forecasting Accuracy (Retail)' },
    { id: 'rossmann', name: 'Rossmann Sales (Store Forecasting)' },
    { id: 'dataco', name: 'DataCo Supply Chain (Logistics)' },
    { id: 'olist', name: 'Olist E-Commerce (Customer & Freight)' }
  ];

  const handleSelect = (e) => setSelected(e.target.value);
  const handleLoad = () => { if (selected) onSelectDataset(selected); };

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: '12px', border: '1px solid var(--primary-color)', boxShadow: '0 0 10px rgba(0, 240, 255, 0.1)' }}>
      <div>
        <h3 style={{ margin: 0, marginBottom: '4px', color: 'var(--primary-color)' }}>Demo Domains</h3>
        <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>Instantly pre-load a massive dataset.</p>
      </div>
      <select 
        value={selected} 
        onChange={handleSelect} 
        disabled={loading}
        style={{ padding: '10px', borderRadius: '4px', background: 'rgba(0,0,0,0.6)', color: 'white', border: '1px solid var(--border-color)', width: '100%', outline: 'none' }}
      >
        <option value="" disabled>Select business domain...</option>
        {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
      </select>
      <button 
        className="btn primary full" 
        onClick={handleLoad} 
        disabled={!selected || loading}
        style={{ position: 'relative' }}
      >
        {loading ? "Preloading Data..." : "Load Domain"}
      </button>
    </div>
  );
}
