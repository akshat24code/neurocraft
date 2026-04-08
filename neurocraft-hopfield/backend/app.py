from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from hopfield_net import HopfieldNetwork, get_letter_patterns

app = Flask(__name__)
CORS(app)

# Initialize and train network
patterns = get_letter_patterns()
train_data = list(patterns.values())
hn = HopfieldNetwork(144) # 12x12
hn.train(train_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    grid = data.get('matrix') # 2D array 12x12
    if not grid:
        return jsonify({"error": "No matrix provided"}), 400
    
    # Flatten grid and convert 0 to -1
    input_vector = np.array(grid).flatten()
    input_vector[input_vector == 0] = -1
    
    # Prediction
    predicted_vector = hn.predict(input_vector)
    
    # Map back to letter
    best_match = "Unknown"
    max_similarity = -1
    
    for letter, pattern in patterns.items():
        similarity = np.dot(predicted_vector, pattern)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = letter
            
    # Reconstructed as matrix
    reconstructed_matrix = predicted_vector.reshape(12, 12).tolist()
    
    return jsonify({
        "letter": best_match,
        "reconstructed": reconstructed_matrix,
        "similarity": float(max_similarity) / 144.0 # Confidence score
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
