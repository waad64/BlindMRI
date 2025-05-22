import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, GRU, RepeatVector, TimeDistributed, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= CONFIG =================
np.random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# =============== DATA LOADING & PREPROCESSING ===============

def load_and_preprocess(filepath, seq_len=100):
    df = pd.read_csv(filepath)
    
    seq_cols = ['EDA', 'IBI_seq', 'HRV']  # sequences
    
    static_cols = [
        'TEMP', 'Patient Age', 'Patient Gender', 'SBP', 'DPB', 'GLU', 'HR', 'MAP', 'PDB', 'NBM',
        'Anxiety_Level', 'Blindness_Duration', 'Cause_Blindness', 'First_MRI_Experience',
        'Headphones_Provided', 'Mobility_Independence'
    ]
    
    def parse_seq(s):
        if isinstance(s, str):
            arr = np.array([float(x) for x in s.split(',')])
            if len(arr) >= seq_len:
                return arr[:seq_len]
            else:
                return np.pad(arr, (0, seq_len - len(arr)), 'constant')
        else:
            return np.zeros(seq_len)
    
    for col in seq_cols:
        df[col] = df[col].apply(parse_seq)
        
    df['Patient Gender'] = df['Patient Gender'].astype(int)
    df['Cause_Blindness'] = df['Cause_Blindness'].astype(int)
    
    static_data = df[static_cols].copy()
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    static_scaled = scaler.fit_transform(static_data)
    
    # Extract gender one-hot from Patient Gender (2 classes)
    gender = df['Patient Gender'].values.reshape(-1, 1)
    gender_onehot = np.eye(2)[gender.flatten()]
    
    gender_idx = static_cols.index('Patient Gender')
    static_scaled = np.delete(static_scaled, gender_idx, axis=1)
    
    X_static = np.hstack([static_scaled, gender_onehot])
    
    X_seq = np.stack([np.stack(df[col].values) for col in seq_cols], axis=0).transpose(1, 2, 0)
    
    y = gender_onehot  # conditioning label
    
    Xs_train, Xs_val, Xseq_train, Xseq_val, y_train, y_val = train_test_split(
        X_static, X_seq, y, test_size=0.2, random_state=42)
    
    return (Xs_train, Xseq_train, y_train), (Xs_val, Xseq_val, y_val), scaler, seq_cols, seq_len

# =============== MODELS ===============

def build_generator(latent_dim, seq_len, n_seq_features, n_static_features, n_classes):
    noise_input = Input(shape=(latent_dim,), name="noise")
    label_input = Input(shape=(n_classes,), name="label")
    
    x = Concatenate()([noise_input, label_input])
    
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Dense(seq_len * 64)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = RepeatVector(seq_len)(x)
    
    x = GRU(128, return_sequences=True)(x)
    x = GRU(64, return_sequences=True)(x)
    
    seq_output = TimeDistributed(Dense(n_seq_features, activation='tanh'), name="seq_output")(x)
    
    static_dense = Dense(128)(Concatenate()([noise_input, label_input]))
    static_dense = LeakyReLU(0.2)(static_dense)
    static_dense = BatchNormalization()(static_dense)
    static_output = Dense(n_static_features, activation='tanh', name="static_output")(static_dense)
    
    generator = Model([noise_input, label_input], [seq_output, static_output], name="generator")
    return generator

def build_discriminator(seq_len, n_seq_features, n_static_features, n_classes):
    seq_input = Input(shape=(seq_len, n_seq_features), name="seq_input")
    static_input = Input(shape=(n_static_features,), name="static_input")
    label_input = Input(shape=(n_classes,), name="label_input")
    
    x_seq = GRU(64, return_sequences=True)(seq_input)
    x_seq = GRU(32)(x_seq)
    
    x = Concatenate()([x_seq, static_input, label_input])
    
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    
    x = Dense(64)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    
    validity = Dense(1, activation='sigmoid')(x)
    
    discriminator = Model([seq_input, static_input, label_input], validity, name="discriminator")
    return discriminator

# =============== TRAINING ===============

def train_cgan(X_static_train, X_seq_train, y_train,
               epochs=2000, batch_size=64,
               latent_dim=100,
               seq_len=100,
               n_seq_features=3,
               n_static_features=19, 
               n_classes=2,
               eval_interval=100):
    
    discriminator = build_discriminator(seq_len, n_seq_features, n_static_features, n_classes)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    
    generator = build_generator(latent_dim, seq_len, n_seq_features, n_static_features, n_classes)
    
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(n_classes,))
    gen_seq, gen_static = generator([noise, label])
    
    discriminator.trainable = False
    validity = discriminator([gen_seq, gen_static, label])
    
    combined = Model([noise, label], validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    
    d_losses, g_losses = [], []
    
    for epoch in tqdm(range(epochs)):
        idx = np.random.randint(0, X_static_train.shape[0], batch_size)
        real_static = X_static_train[idx]
        real_seq = X_seq_train[idx]
        real_labels = y_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_labels = y_train[np.random.randint(0, y_train.shape[0], batch_size)]
        gen_seq, gen_static = generator.predict([noise, sampled_labels], verbose=0)
        
        d_loss_real = discriminator.train_on_batch([real_seq, real_static, real_labels], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([gen_seq, gen_static, sampled_labels], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        g_loss = combined.train_on_batch([noise, sampled_labels], np.ones((batch_size, 1)))
        
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        
        if epoch % eval_interval == 0:
            print(f"Epoch {epoch} | D loss: {d_loss[0]:.4f} | D acc: {d_loss[1]:.4f} | G loss: {g_loss:.4f}")
    
    return generator, discriminator

# =============== SAMPLE GENERATION ===============

def generate_samples(generator, latent_dim, n_samples, n_classes=2):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    labels = np.eye(n_classes)[np.random.randint(0, n_classes, n_samples)]
    seq_generated, static_generated = generator.predict([noise, labels])
    return seq_generated, static_generated, labels

# =============== POSTPROCESSING ===============

def postprocess_and_save(seq_gen, static_gen, labels_gen, scaler, seq_cols,
                         output_csv_path="dataset_v3.csv"):
    gender_onehot = static_gen[:, -2:]
    static_scaled = static_gen[:, :-2]
    
    static_original = scaler.inverse_transform(static_scaled)
    
    gender_labels = np.argmax(gender_onehot, axis=1)
    
    static_cols = scaler.feature_names_in_.tolist()
    gender_idx = static_cols.index('Patient Gender')
    
    static_df = pd.DataFrame(static_original, columns=[col for col in static_cols if col != 'Patient Gender'])
    static_df.insert(gender_idx, 'Patient Gender', gender_labels)
    
    categorical_cols = ['Anxiety_Level', 'Blindness_Duration', 'Cause_Blindness', 'First_MRI_Experience',
                        'Headphones_Provided', 'Mobility_Independence']
    for col in categorical_cols:
        static_df[col] = static_df[col].round().astype(int)
    
    seq_dict = {}
    for i, col in enumerate(seq_cols):
        seq_dict[col] = [','.join(map(str, seq_gen[sample, :, i])) for sample in range(seq_gen.shape[0])]
    seq_df = pd.DataFrame(seq_dict)
    
    final_df = pd.concat([static_df, seq_df], axis=1)
    final_df.to_csv(output_csv_path, index=False)
    print(f"[+] Synthetic blind dataset saved to {output_csv_path}")

# =============== MAIN ===============

if __name__ == "__main__":
    data_path = "training/csv/dataset_v2_merged.csv"
    
    (X_static_train, X_seq_train, y_train), (X_static_val, X_seq_val, y_val), scaler, seq_cols, seq_len = load_and_preprocess(data_path)
    
    latent_dim = 100
    n_seq_features = len(seq_cols)
    n_static_features = X_static_train.shape[1]
    n_classes = 2
    
    generator, discriminator = train_cgan(
        X_static_train, X_seq_train, y_train,
        epochs=2000,
        batch_size=64,
        latent_dim=latent_dim,
        seq_len=seq_len,
        n_seq_features=n_seq_features,
        n_static_features=n_static_features,
        n_classes=n_classes,
        eval_interval=100
    )
    
    seq_gen, static_gen, labels_gen = generate_samples(generator, latent_dim, 1000, n_classes)
    
    postprocess_and_save(seq_gen, static_gen, labels_gen, scaler, seq_cols)
