import streamlit as st

#EDA pkg
import pandas as pd
import numpy as np

# Model Load/Save
from joblib import load
import joblib
import os

#Data Viz pkg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns



#### Functions

## Load css
def load_css(css_name):
	with open(css_name) as f:
		st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

## Load icon
def load_icon(name):
	st.markdown('<i class ="material-icons">{}</i>'.format(name), unsafe_allow_html=True)

## remote_css
def remote_css(url):
    st.markdown('<style src="{}"></style>'.format(url), unsafe_allow_html=True)

## icon-css
def icon_css(icone_name):
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


## Load Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# Main
def main():
	""" Percentage Score Prediction ML App with Streamlit """
	st.title("Percentage Score Prediction")
	st.text("ML Prediction App with Streamlit")

	# Loading Dataset
	url='http://bit.ly/w-data'
	df=pd.read_csv(url)
	
	# Sidebar (TABS/ Menus)
	bars = ['EDA','Prediction','About']
	choice = st.sidebar.selectbox("Choose Activity", bars)


	# Choice EDA
	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")
		load_css('icon.css') #function defines at the top
		load_icon('dashboard') #function defines at the top

		if st.checkbox("Show Dataset Preview"):
			num = st.number_input("Enter Number of Rows to Preview: ", value=5)
			st.dataframe(df.head(num))

		if st.checkbox("Shape of Dataset"):
			st.write(df.shape)
			dim = st.radio("Show Dimensions by :",("Rows","Columns"))

			if dim == "Rows":
				st.text("Number of Rows :")
				st.write(df.shape[0])
			elif dim == "Columns":
				st.text("Number of Columns :")
				st.write(df.shape[1])

		if st.checkbox("Column Names"):
			all_columns = df.columns.tolist()
			st.write(all_columns)

		if st.checkbox("Select Columns to Show"):
			all_columns = df.columns.tolist()
			selected_col = st.multiselect("Select Columns", all_columns)
			new_col = df[selected_col]
			st.dataframe(new_col)

		if st.checkbox("Show Info"):
			st.write(df.dtypes)

		if st.checkbox("Show Description"):
			st.write(df.describe())

		st.subheader("Data Visualization")
		load_css('icon.css')
		load_icon('show_charts')

		# Pair plot
		if st.checkbox("Pair Plot"):
			st.write(sns.pairplot(df))
			st.pyplot()

		# Dist plot
		if st.checkbox("Dist Plot"):
			dim1 = st.radio("Show Dist Plot by :",("Hours","Scores"))

			if dim1 == "Hours":
				sns.set_color_codes()
				st.write(sns.distplot(df['Hours'], color="r"))
				st.pyplot()
			elif dim1 == "Scores":
				sns.set_color_codes()
				st.write(sns.distplot(df['Scores'], color="g"))
				st.pyplot()

		# Data Plotting
		if st.checkbox("Plot Data"):
			df.plot(x = 'Hours',y = 'Scores', style='*')
			plt.xlabel('Hours studied')
			plt.ylabel('Percentage Score')
			plt.title('Hours vs percentage Score')
			st.pyplot()

		# Line plot with Seaborn
		if st.checkbox("Line Plot"):
			st.write(sns.lineplot(df['Hours'],df['Scores']))
			st.pyplot()





		# Correlation plot with Matplotlib
		if st.checkbox("Correlation Plot [using Matplotlib]"):
			plt.matshow(df.corr())
			st.pyplot()

		# Correlation plot with Seaborn
		if st.checkbox("Correlation Plot with Annotation [using Seaborn]"):
			st.write(sns.heatmap(df.corr(), annot=True))
			st.pyplot()

	
	# CHOICE FOR Prediction
	if choice == 'Prediction':
		st.subheader("Prediction of Scores based on Study Hours")
		st.markdown('<style>' + open('icon.css').read() + '</style>', unsafe_allow_html=True)
		st.markdown('<i class="material-icons">mood</i>', unsafe_allow_html=True)
		
		load_css('icon.css') # function defines at the top
		# load_icon('timeline') #function defines at the top

		load_icon('work')
		hours = float(st.number_input("'Enter Studied hours to predict your score ",1.0000,10.0000))
		
		## Show Summary
		selected_columns = [hours]
		vectorized_result = [hours]
		vectorization = np.array(vectorized_result).reshape(1,-1)
			
		if st.checkbox("Show Summary"):
			st.info (selected_columns)

			if st.checkbox("Summary in JSON Format"):
				simple_result = {"hours":hours}

				st.subheader("Pretified Result in JSON")
				st.json(simple_result)

			if st.checkbox("Summary in Encoded Format"):
				st.text("Using Encoding for Prediction")
				st.success(vectorized_result)


		# Making Predictions
		st.write("")
		load_icon('show_charts')

		st.subheader("Prediction")
		if st.checkbox("Make Prediction"):
			if st.button("Predict"):
				model_predictor = load_model("models/pred_score_l_reg_model.pkl")
				prediction = model_predictor.predict(vectorization)
				st.success("Predicted Score based on your study hours is :: {}%".format(round(prediction[0]),2))

	if choice == 'About':
		st.subheader("About")
		st.markdown("""
			#### Task - 1: Percentage Score Predictor ML Web App 
			##### Built with Streamlit
			#### 
			
			#### Company
			+ GRIP - The Spark Foundation 

			#### By
			+ Hrishikesh Sharad Malkar
			

			""")


if __name__ == '__main__':
	main()

