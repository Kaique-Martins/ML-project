import argparse
import os
from pathlib import Path
import joblib
import numpy as np
from src.data import load_csv, preprocess
from src.model import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to csv file')
    parser.add_argument('--target', required=True, help='target column name')
    parser.add_argument('--model-dir', default='models/my_model', help='where to save the model')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_csv(args.data, args.target)
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)

    input_shape = X_train_s.shape[1]
    n_classes = 1 if len(np.unique(y_train)) == 2 else len(np.unique(y_train))
    model = build_model(input_shape, n_classes=n_classes)

    model.fit(X_train_s, y_train, validation_data=(X_test_s, y_test), epochs=args.epochs, batch_size=32)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir)

    # save scaler
    joblib.dump(scaler, model_dir / 'scaler.joblib')
    print('Model and scaler saved to', model_dir)

if __name__ == '__main__':
    main()
