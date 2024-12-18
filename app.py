from flask import Flask, render_template, request, jsonify
from pykrige.ok import OrdinaryKriging
from folium.plugins import HeatMap
from flask_wtf.csrf import CSRFProtect
import pandas as pd
import folium
import os
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired


app = Flask(__name__)
app.secret_key = os.urandom(24)
csrf = CSRFProtect(app)

# Membuat form dengan WTForms
class PredictionForm(FlaskForm):
    latitude = StringField('Latitude', validators=[DataRequired()])
    longitude = StringField('Longitude', validators=[DataRequired()])

# Model training (gunakan data yang sama)
train_data = pd.read_csv('train_data.csv')
#test_data = pd.read_excel('test_data.xlsx')
dlkh_data = pd.read_csv('train_data.csv')

# Predict function for multiple features
def kriging_predict_multiple(lat, lon, features):
    predictions = {}
    for feature in features:
        OK = OrdinaryKriging(
            train_data['Latitude'], 
            train_data['Longitude'], 
            train_data[feature],
            variogram_model='spherical',
            verbose=False,
            enable_plotting=False
        )
        prediction, _ = OK.execute('points', lat, lon)
        predictions[feature] = round(prediction[0], 2)
    return predictions

def kriging_predict_multiple_dlkh(lat, lon, features_dlkh):
    predictions_dlkh = {}
    for feature_dlkh in features_dlkh:
        OK_dlkh = OrdinaryKriging(
            dlkh_data['Latitude'], 
            dlkh_data['Longitude'], 
            dlkh_data[feature_dlkh],
            variogram_model='gaussian',
            verbose=False,
            enable_plotting=False
        )
        prediction_dlkh, _ = OK_dlkh.execute('points', lat, lon)
        predictions_dlkh[feature_dlkh] = round(prediction_dlkh[0], 2)
    return predictions_dlkh

# Route for prediction form
@app.route('/', methods=['GET', 'POST'])
def form():
    form = PredictionForm()
    if form.validate_on_submit():
        latitude = form.latitude.data
        longitude = form.longitude.data
        features = request.form['features'].split(',')
        # Proses prediksi
        predictions = kriging_predict_multiple(latitude, longitude, features)
        return render_template('result_with_map.html', predictions=predictions)
    return render_template('form.html', form=form)

# Route for prediction DLH form
@app.route('/form_dlh', methods=['GET', 'POST'])
def form_dlh():
    form_dlh = PredictionForm()
    if form_dlh.validate_on_submit():
        latitude = form_dlh.latitude.data
        longitude = form_dlh.longitude.data
        features_dlkh = request.form['features_dlkh'].split(',')
        # Proses prediksi
        predictions_dlkh = kriging_predict_multiple_dlkh(latitude, longitude, features_dlkh)
        return render_template('result_with_map.html', predictions_dlkh=predictions_dlkh)
    return render_template('form_dlh.html', form_dlh=form_dlh)



@app.route('/predict_form', methods=['POST', 'GET'])
def predict_form():
    form = PredictionForm()

    if form.validate_on_submit():
        try:
            latitude = float(form.latitude.data)
            longitude = float(form.longitude.data)
            features = request.form['features'].split(',')

            # Validasi latitude dan longitude
            if not latitude or not longitude:
                raise ValueError("Latitude dan longitude tidak boleh kosong.")
            latitude = float(latitude)
            longitude = float(longitude)

            if not (-90 <= latitude <= 90):
                raise ValueError("Latitude harus berada di antara -90 dan 90.")
            if not (-180 <= longitude <= 180):
                raise ValueError("Longitude harus berada di antara -180 dan 180.")

            # Validasi fitur
            if not features or all(f.strip() == '' for f in features):
                raise ValueError("Features tidak boleh kosong.")

            # Get predictions
            predictions = kriging_predict_multiple(latitude, longitude, features)

            # Create map with Folium
            m = folium.Map(location=[latitude, longitude], zoom_start=12)

            # Add CircleMarker for training data
            for _, row in train_data.iterrows():
                location_name = row.get('LocationName', 'Unknown Location')  # Kolom nama lokasi
                popup_content = f"""
                <b>Nama Lokasi:</b> {location_name}<br>
                <b>Latitude:</b> {row['Latitude']}<br>
                <b>Longitude:</b> {row['Longitude']}
                """
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=7,  # Ukuran lingkaran
                    color="blue",  # Warna lingkaran
                    fill=True,
                    fill_color="blue",  # Warna isi lingkaran
                    fill_opacity=0.7,  # Transparansi isi
                    popup=popup_content  # Informasi popup
                ).add_to(m)

            # Add Marker for user input location
            popup_info = f"Latitude: {latitude}, Longitude: {longitude}<br>Predictions:<br>"
            for feature, value in predictions.items():
                popup_info += f"{feature}: {value:.2f}<br>"

            folium.Marker(
                location=[latitude, longitude],
                radius=10,
                popup=popup_info,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)

            # Save map as HTML file
            map_file = 'static/map.html'
            m.save(map_file)

            # Render results page with map
            return render_template('result_with_map.html', latitude=latitude, longitude=longitude, predictions=predictions, map_file=map_file)

        except ValueError as ve:
            # Tangkap error validasi dan kembalikan pesan error ke pengguna
            return render_template('form.html', form=form, error=str(ve))
        
        except Exception as e:
            # Tangkap error umum lainnya
            return render_template('form.html', form=form, error="Terjadi kesalahan: " + str(e))

    return render_template('form.html', form=form)

@app.route('/predict_form2', methods=['POST', 'GET'])
def predict_form2():
    form_dlh = PredictionForm()

    if form_dlh.validate_on_submit():
        try:
            latitude = float(form_dlh.latitude.data)
            longitude = float(form_dlh.longitude.data)
            features = request.form['features_dlkh'].split(',')

            # Validasi latitude dan longitude
            if not latitude or not longitude:
                raise ValueError("Latitude dan longitude tidak boleh kosong.")
            latitude = float(latitude)
            longitude = float(longitude)

            if not (-90 <= latitude <= 90):
                raise ValueError("Latitude harus berada di antara -90 dan 90.")
            if not (-180 <= longitude <= 180):
                raise ValueError("Longitude harus berada di antara -180 dan 180.")

            # Validasi fitur
            if not features or all(f.strip() == '' for f in features):
                raise ValueError("Features tidak boleh kosong.")

            # Get predictions
            predictions_dlkh = kriging_predict_multiple_dlkh(latitude, longitude, features)

            # Create map with Folium
            m = folium.Map(location=[latitude, longitude], zoom_start=12)

            # Add CircleMarker for training data
            for _, row in train_data.iterrows():
                location_name = row.get('LocationName', 'Unknown Location')  # Kolom nama lokasi
                popup_content = f"""
                <b>Nama Lokasi:</b> {location_name}<br>
                <b>Latitude:</b> {row['Latitude']}<br>
                <b>Longitude:</b> {row['Longitude']}
                """
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=7,  # Ukuran lingkaran
                    color="blue",  # Warna lingkaran
                    fill=True,
                    fill_color="blue",  # Warna isi lingkaran
                    fill_opacity=0.7,  # Transparansi isi
                    popup=popup_content  # Informasi popup
                ).add_to(m)

            # Add Marker for user input location
            popup_info = f"Latitude: {latitude}, Longitude: {longitude}<br>Predictions:<br>"
            for feature, value in predictions_dlkh.items():
                popup_info += f"{feature}: {value:.2f}<br>"

            folium.Marker(
                location=[latitude, longitude],
                radius=10,
                popup=popup_info,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)

            # Save map as HTML file
            map_file = 'static/map.html'
            m.save(map_file)

            # Render results page with map
            return render_template('resultdlh_with_map.html', latitude=latitude, longitude=longitude, predictions_dlkh=predictions_dlkh, map_file=map_file)

        except ValueError as ve:
            # Tangkap error validasi dan kembalikan pesan error ke pengguna
            return render_template('form_dlh.html', form_dlh=form_dlh, error=str(ve))
        
        except Exception as e:
            # Tangkap error umum lainnya
            return render_template('form_dlh.html', form_dlh=form_dlh, error="Terjadi kesalahan: " + str(e))

    return render_template('form_dlh.html', form_dlh=form_dlh)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=8080)
