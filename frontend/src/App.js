//App.js

import React, { useState } from 'react';
import Record from './Record';
import Result from './Result';
import './App.css';

function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="container">
      <h1>Speaker Attribute Predictor</h1>
      <Record setResult={setResult} />
      {result && <Result result={result} />}
    </div>
  );
}

export default App;
