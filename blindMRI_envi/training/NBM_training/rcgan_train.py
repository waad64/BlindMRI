import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, GRU, LSTM, RepeatVector, TimeDistributed, Dropout, LeakyReLU, BatchNormalization, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===================== CONFIG =====================
np.random.seed(42)
tf.random.set_seed(42)

# GPU config (optional)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# =============== DATA LOADING & PREPROCESSING ===============

def load_and_preprocess(filepath, seq_len=100):
    df = pd.read_csv(filepath)
    
    seq_cols = ['EDA', 'IBI_seq', 'HRV']  # sequential, 100 steps each
    static_cols = ['TEMP', 'Patient Age', 'Patient Gender', 'SBP', 'DPB', 'GLU', 'HR', 'MAP', 'PDB','NBM']
    
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
        
    scaler = MinMaxScaler(feature_range=(-1, 1))
    static_data = df[static_cols].copy()
    static_data['Patient Gender'] = static_data['Patient Gender'].astype(int)
    
    gender = static_data['Patient Gender'].values.reshape(-1, 1)
    static_data = static_data.drop(columns=['Patient Gender'])
    static_scaled = scaler.fit_transform(static_data)
    
    gender_onehot = np.eye(2)[gender.flatten()]
    X_static = np.hstack([static_scaled, gender_onehot])
    
    # Fixed here: use a list comprehension inside np.stack, specify axis=0
    X_seq = np.stack([np.stack(df[col].values) for col in seq_cols], axis=0).transpose(1, 2, 0)
    
    y = gender_onehot
    
    Xs_train, Xs_val, Xseq_train, Xseq_val, y_train, y_val = train_test_split(
        X_static, X_seq, y, test_size=0.2, random_state=42)
    
    return (Xs_train, Xseq_train, y_train), (Xs_val, Xseq_val, y_val), scaler, seq_cols, seq_len


# ===================== RCGAN MODEL COMPONENTS =====================

def build_generator(latent_dim=100, seq_len=100, n_seq_features=3, n_static_features=10, n_classes=2):
    # Latent noise + conditioning label
    noise_input = Input(shape=(latent_dim,), name="noise_input")
    label_input = Input(shape=(n_classes,), name="label_input")
    
    # Combine noise + label
    x = Concatenate()([noise_input, label_input])
    
    # Dense layers to expand noise to latent space for sequence generation
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    # Repeat vector to create initial sequence seed
    x = Dense(seq_len * 64)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = RepeatVector(seq_len)(x)  # shape: (seq_len, 64)
    
    # RNN layers to generate sequence features
    x = GRU(128, return_sequences=True)(x)
    x = GRU(64, return_sequences=True)(x)
    
    # Generate sequence output: EDA, IBI_seq, HRV (3 features)
    seq_output = TimeDistributed(Dense(n_seq_features, activation='tanh'), name="sequence_output")(x)
    
    # Generate static features output
    static_dense = Dense(128)(Concatenate()([noise_input, label_input]))
    static_dense = LeakyReLU(0.2)(static_dense)
    static_dense = BatchNormalization()(static_dense)
    static_output = Dense(n_static_features, activation='tanh', name="static_output")(static_dense)
    
    generator = Model(inputs=[noise_input, label_input], outputs=[seq_output, static_output], name="RCGAN_Generator")
    return generator

def build_discriminator(seq_len=100, n_seq_features=3, n_static_features=10, n_classes=2):
    # Sequence input
    seq_input = Input(shape=(seq_len, n_seq_features), name="seq_input")
    # Static input
    static_input = Input(shape=(n_static_features,), name="static_input")
    # Label input for conditioning
    label_input = Input(shape=(n_classes,), name="label_input")
    
    # Process sequences through RNN
    x_seq = GRU(64, return_sequences=True)(seq_input)
    x_seq = GRU(32)(x_seq)  # shape: (batch_size, 32)
    
    # Concatenate sequence output + static features + labels
    x = Concatenate()([x_seq, static_input, label_input])
    
    # Dense layers for discrimination
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    
    x = Dense(64)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.4)(x)
    
    validity = Dense(1, activation='sigmoid')(x)
    
    discriminator = Model(inputs=[seq_input, static_input, label_input], outputs=validity, name="RCGAN_Discriminator")
    return discriminator

# ===================== TRAINING LOOP =====================

def train_rcgan(
    X_static_train, X_seq_train, y_train,
    epochs=2000, batch_size=32,
    latent_dim=100,
    seq_len=100,
    n_seq_features=3,
    n_static_features=10,
    n_classes=2,
    eval_interval=100):

    discriminator = build_discriminator(seq_len, n_seq_features, n_static_features, n_classes)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    
    generator = build_generator(latent_dim, seq_len, n_seq_features, n_static_features, n_classes)
    
    # GAN combined model
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(n_classes,))
    gen_seq, gen_static = generator([noise, label])
    
    discriminator.trainable = False
    validity = discriminator([gen_seq, gen_static, label])
    
    combined = Model([noise, label], validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    
    d_losses, g_losses = [], []
    
    for epoch in tqdm(range(epochs)):
        # Sample real data batch
        idx = np.random.randint(0, X_static_train.shape[0], batch_size)
        real_static = X_static_train[idx]
        real_seq = X_seq_train[idx]
        real_labels = y_train[idx]
        
        # Generate fake data batch
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_labels = y_train[np.random.randint(0, y_train.shape[0], batch_size)]
        gen_seq, gen_static = generator.predict([noise, sampled_labels], verbose=0)
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch([real_seq, real_static, real_labels], np.ones((batch_size,1)))
        d_loss_fake = discriminator.train_on_batch([gen_seq, gen_static, sampled_labels], np.zeros((batch_size,1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator (via combined model)
        g_loss = combined.train_on_batch([noise, sampled_labels], np.ones((batch_size, 1)))
        
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        
        if epoch % eval_interval == 0:
            print(f"Epoch {epoch} | D loss: {d_loss[0]:.4f} | D acc: {d_loss[1]:.4f} | G loss: {g_loss:.4f}")
            
    return generator, discriminator

# ===================== FINAL SAMPLE GENERATION =====================

def generate_samples(generator, latent_dim, n_samples, n_classes=2):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    labels = np.eye(n_classes)[np.random.randint(0, n_classes, n_samples)]
    seq_generated, static_generated = generator.predict([noise, labels])
    return seq_generated, static_generated, labels

# ===================== MAIN =====================

if __name__ == "__main__":
    data_path = "PDB_training/csv/llm_generate_dataset_with_pdb.csv"
    
    (X_static_train, X_seq_train, y_train), (X_static_val, X_seq_val, y_val), scaler, seq_cols, seq_len = load_and_preprocess(data_path)
    
    latent_dim = 100
    n_seq_features = len(seq_cols)
    n_static_features = X_static_train.shape[1]
    n_classes = 2
    
    generator, discriminator = train_rcgan(
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
    
    # Postprocess and save your generated data here (like inverse scaling static features etc.)

def postprocess_and_save(seq_gen, static_gen, labels_gen, scaler, seq_cols, output_csv_path="generated_data.csv"):
    # Inverse scale static features (except gender, which was one-hot encoded)
    # static_gen shape: (n_samples, n_static_features) = scaled + one-hot gender
    
    # Separate gender one-hot (last two cols)
    gender_onehot = static_gen[:, -2:]
    static_scaled = static_gen[:, :-2]
    
    # Inverse transform static scaled features
    static_original = scaler.inverse_transform(static_scaled)
    
    # Convert gender one-hot back to label
    gender_labels = np.argmax(gender_onehot, axis=1)
    
    # Compose static DataFrame
    static_df = pd.DataFrame(static_original, columns=[col for col in scaler.feature_names_in_])
    static_df['Patient Gender'] = gender_labels
    
    # Sequence data shape: (samples, seq_len, n_seq_features)
    # Convert sequence arrays to comma-separated strings per sample per feature
    seq_dict = {}
    for i, col in enumerate(seq_cols):
        seq_dict[col] = [','.join(map(str, seq_gen[sample, :, i])) for sample in range(seq_gen.shape[0])]
    seq_df = pd.DataFrame(seq_dict)
    
    # Combine all into one DataFrame
    final_df = pd.concat([static_df, seq_df], axis=1)
    
    # Save to CSV
    final_df.to_csv(output_csv_path, index=False)
    print(f"[+] Synthetic data saved to {output_csv_path}")

# Example usage after generation
postprocess_and_save(seq_gen, static_gen, labels_gen, scaler, seq_cols)
