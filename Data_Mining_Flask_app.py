from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
# Your API definition
app = Flask(__name__)
@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query=pd.DataFrame(json_)
            
            for c in query:
                df_fq_map=query[c].value_counts().to_dict()
                query[c]=query[c].map(df_fq_map)
            'STOLEN', 'RECOVERED', 'UNKNOWN'
            prediction = list(lr.predict(query))
            print({'prediction': str(prediction)})
            convert_for_user = {
                    "[20929]": "STOLEN",
                    "[403]": "UNKNOWN",
                    "[252]": "RECOVERED"}
            prediction=str(prediction)
            result=convert_for_user[prediction]
            return jsonify({'prediction': result})
            return "Welcome to Bike Data model APIs!"

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12346 # If you don't provide any port the port will be set to 12345

    lr = joblib.load("F:\DataMiningProject\modelx.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("F:\\DataMiningProject\\model_columnsx.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=True)
