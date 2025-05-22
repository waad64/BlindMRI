import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# ====================== Configs ======================
np.random.seed(42)
tf.random.set_seed(42)

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Directory structure
for folder in ['plots', 'csv', 'generated_samples']:
    os.makedirs(folder, exist_ok=True)

# ====================== Data Loading & Preprocessing ======================
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    seq_cols = ['EDA', 'IBI_seq', 'HRV']
    seq_len = 100

    def parse_sequence(seq):
        if isinstance(seq, str):
            return np.array([float(x) for x in seq.split(',')])
        return np.full(seq_len, 0.0)

    # Format sequence features
    for col in seq_cols:
        df[col] = df[col].apply(parse_sequence)
        df[col] = df[col].apply(lambda x: x[:seq_len] if len(x) > seq_len else np.pad(x, (0, seq_len - len(x)), 'constant'))

    static_cols = ['TEMP', 'Patient Age', 'SBP', 'DPB', 'GLU', 'HR', 'MAP']
    static_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_static = static_scaler.fit_transform(df[static_cols])

    X_seq = np.concatenate([np.stack(df[col].values) for col in seq_cols], axis=1)

    # Z-score normalization of sequences
    X_seq = (X_seq - X_seq.mean(axis=1, keepdims=True)) / (X_seq.std(axis=1, keepdims=True) + 1e-8)

    X = np.concatenate([X_static, X_seq], axis=1)

    # Labels
    y = df['Patient Gender'].astype(int).values.reshape(-1, 1)
    y_onehot = np.eye(2)[y.flatten()]

    return train_test_split(X, y_onehot, test_size=0.2, random_state=42), static_scaler, seq_cols, seq_len

# ====================== Model Definitions ======================
def build_generator(latent_dim=100, n_classes=2, output_dim=None):
    """Build the generator model with conditional input."""
    input_shape = latent_dim + n_classes

    input_layer = Input(shape=(input_shape,))
    x = Dense(512)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Dense(2048)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Dense(output_dim, activation='tanh')(x)

    model = Model(inputs=input_layer, outputs=x, name="Generator")

    # Inputs for noise and label
    noise_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(n_classes,))
    merged_input = Concatenate()([noise_input, label_input])
    output = model(merged_input)

    return Model([noise_input, label_input], output, name="ConditionalGenerator")


def build_discriminator(input_dim):
    model = Sequential([
        Dense(2048, input_dim=input_dim + 2),
        LeakyReLU(0.2),
        Dropout(0.4),
        Dense(1024),
        LeakyReLU(0.2),
        Dropout(0.4),
        Dense(512),
        LeakyReLU(0.2),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    sample = Input(shape=(input_dim,))
    label = Input(shape=(2,))
    combined = Concatenate()([sample, label])
    validity = model(combined)
    return Model([sample, label], validity, name="Discriminator")

# ====================== Training Logic ======================
def generate_samples(generator, n, latent_dim):
    noise = np.random.normal(0, 1, (n, latent_dim))
    labels = np.random.randint(0, 2, (n, 1))
    labels_onehot = np.eye(2)[labels.flatten()]
    return generator.predict([noise, labels_onehot]), labels

def train_cgan(X_train, y_train, epochs=2000, batch_size=32, eval_interval=50, early_stop_threshold=0.02):
    latent_dim = 100
    n_classes = 2
    input_dim = X_train.shape[1]

    generator = build_generator(latent_dim=latent_dim, n_classes=n_classes, output_dim=input_dim)
    discriminator = build_discriminator(input_dim=input_dim)
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, decay=1e-6), metrics=['accuracy'])



    noise_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(2,))
    generated = generator([noise_input, label_input])
    discriminator.trainable = False
    validity = discriminator([generated, label_input])
    combined = Model([noise_input, label_input], validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, decay=1e-6))

    d_losses, g_losses = [], []

    for epoch in tqdm(range(epochs), desc="Training"):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_samples = X_train[idx]
        real_labels = y_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_labels = np.eye(2)[np.random.randint(0, 2, batch_size)]
        fake_samples = generator.predict([noise, gen_labels], verbose=0)

        d_loss_real = discriminator.train_on_batch([real_samples, real_labels], np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch([fake_samples, gen_labels], np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        sampled_labels = np.eye(2)[np.random.randint(0, 2, batch_size)]
        g_loss = combined.train_on_batch([noise, sampled_labels], np.ones((batch_size, 1)))

        d_losses.append(d_loss[0])
        g_losses.append(g_loss)

        if epoch % eval_interval == 0:
            gen_samples, _ = generate_samples(generator, 1000, latent_dim)
            wd = np.mean([wasserstein_distance(X_train[:, i], gen_samples[:, i]) for i in range(10)])

            plt.plot(d_losses, label='D loss')
            plt.plot(g_losses, label='G loss')
            plt.legend(); plt.title(f"Losses at Epoch {epoch}")
            plt.savefig(f"plots/loss_epoch_{epoch}.png")
            plt.close()

            print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f} | Wasserstein: {wd:.4f}")

            if d_loss[0] < early_stop_threshold:
                print(f"Stopped early at epoch {epoch}")
                break

    return generator

# ====================== Final Sample Generation ======================
def save_generated_data(generator, latent_dim, scaler, seq_cols, seq_len, original_csv):
    final_samples, final_labels = generate_samples(generator, 2000, latent_dim)
    static_dim = len(['TEMP', 'Patient Age', 'SBP', 'DPB', 'GLU', 'HR', 'MAP'])
    X_static = scaler.inverse_transform(final_samples[:, :static_dim])
    X_seq = final_samples[:, static_dim:]

    original_df = pd.read_csv(original_csv)
    original_seqs = {
        col: np.stack(original_df[col].apply(lambda x: np.array([float(i) for i in str(x).split(',')[:seq_len]]) if isinstance(x, str) else np.zeros(seq_len)))
        for col in seq_cols
    }

    seq_splits = np.split(X_seq, len(seq_cols), axis=1)
    final_seq = {}
    for i, col in enumerate(seq_cols):
        mean = original_seqs[col].mean(axis=1, keepdims=True)
        std = original_seqs[col].std(axis=1, keepdims=True)
        final_seq[col] = seq_splits[i] * std + mean

    df = pd.DataFrame({
        'EDA': [','.join(map(str, s)) for s in final_seq['EDA']],
        'TEMP': X_static[:, 0],
        'Patient Age': X_static[:, 1],
        'Patient Gender': final_labels.flatten(),
        'SBP': X_static[:, 2],
        'DPB': X_static[:, 3],
        'GLU': X_static[:, 4],
        'HR': X_static[:, 5],
        'IBI_seq': [','.join(map(str, s)) for s in final_seq['IBI_seq']],
        'MAP': X_static[:, 6],
        'HRV': [','.join(map(str, s)) for s in final_seq['HRV']]
    })

    df.to_csv("dataset_v1.csv", index=False)
    

# ====================== Main ======================
if __name__ == "__main__":
    (X_train, X_val, y_train, y_val), scaler, seq_cols, seq_len = load_and_preprocess_data("training/csv/llm_generate_dataset.csv")
    generator = train_cgan(X_train, y_train)
    save_generated_data(generator, 100, scaler, seq_cols, seq_len, "training/csv/llm_generate_dataset.csv")
