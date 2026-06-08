import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # BARIS EXPERIMENT DIHAPUS/DIMATIKAN BIAR GAK BENTROK ID DI GITHUB ACTIONS
    # mlflow.set_experiment("Eksperimen_Model_Tanah_Aliya")

    print("Memuat data...")
    # Menggunakan os.path biar Python nyari lokasi file CSV tepat di sebelah file modelling.py ini berada
    import os
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "crop_preprocessed.csv")
    df = pd.read_csv(csv_path)

    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TAMBAHKAN nested=True BIAR SINKRON SAMA RUN ID DARI SERVER GITHUB ACTIONS
    with mlflow.start_run(run_name="RandomForest_Manual_Run", nested=True):
        print("Melatih model Random Forest...")
        
        n_estimators = 100
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        # LOGGING MANUAL
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model_tanah_aliya")
        
        # BARIS SAVE_MODEL DIHAPUS KARENA BIKIN PROSES RE-TRAINING MANDALIK EROR DI SERVER GITHUB
        # mlflow.sklearn.save_model(model, "my_model")
        
        print(f"Model berhasil dilatih! Akurasi: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()