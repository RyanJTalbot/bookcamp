# Football-Price-Prediction-System
This project takes the FIFA dataset and uses the attributes in FIFA to determine the price of a football player. Data is also scraped from Transfermarkt to take the football players football valuation and is used to identify the most important FIFA attributes when determining the price of a football player. The proof of concept website uses only the fifa data and fifa valuations to predict the price of a football player however, models have been ran on the utilise the transfermarkt data also. This can be found within the transfermarkt_models.ipynb file. 

# Tech Stack and Packages
- Python
- Streamlit
- Keras
- Pandas
- Shap
- Sklearn

# Setup and Running
- Install streamlit onto your local machine 
- Download or pull the repo to local device
- Open terminal and go to the repository local
- run the line "streamlit run ProofOfConceptWebApplication.py"
- the local server will run and will create a new tab on your browser (allow for some time for the models to run during startup)
