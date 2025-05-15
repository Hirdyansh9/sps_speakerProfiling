import React, { useState } from 'react';
import Record from './Record';
import Result from './Result';

function App() {
  const [result, setResult] = useState(null);

  return (
    <div style={{ padding: '20px' }}>
      <h1>Speaker Attribute Predictor</h1>
      <Record setResult={setResult} />
      {result && <Result result={result} />}
    </div>
  );
}

export default App;
