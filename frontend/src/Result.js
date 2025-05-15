import React from 'react';

function Result({ result }) {
  return (
    <div>
      <h2>Prediction</h2>
      <p><strong>Age:</strong> {result.age}</p>
      <p><strong>Height:</strong> {result.height} cm</p>
      <p><strong>Gender:</strong> {result.Gender}</p>
    </div>
  );
}

export default Result;
