# PCOS

ML project for Polycystic ovary syndrome. That involves infrequent, irregular or prolonged menstrual periods

# INSTRUCCIONES

Activar el entorno virtual:

.\.venv\Scripts\Activate.ps1

Instalar dependencias:

pip install -r requirements.txt

Ejecutar el modelo:

python train.py

Ejecutar con diferentes hiperparámetros:

python train.py --C 0.1 --max-iter 500 --solver liblinear --random-state 42
