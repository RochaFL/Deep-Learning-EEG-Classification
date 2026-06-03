

import os
# -------------- SEED e ambiente determinístico --------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import random, time, glob
import numpy as np
import mne
import tensorflow as tf

# seeds
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
GLOBAL_RNG = np.random.RandomState(SEED)

# reduce TF threading nondeterminism
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except Exception:
    pass

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Flatten,
                                     Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     AveragePooling2D, Reshape, Activation, InputLayer,
                                     SpatialDropout2D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from sklearn.utils.class_weight import compute_class_weight

# --------- CONFIG ----------
DATA_PATH = 'dados/'
OUTPUT_FILE = 'resultados_pipeline_fix.txt'

CLASSES = ['left_hand', 'right_hand', 'feet', 'tongue']
N_CLASSES = len(CLASSES)
N_SPLITS = 5

F_MIN = 8.0; F_MAX = 30.0
EPOCH_T_MIN = -1.0; EPOCH_T_MAX = 4.0
BASELINE_T_MIN = -1.0; BASELINE_T_MAX = 0.0
CROP_T_MIN = 1.0; CROP_T_MAX = 4.0

BATCH_SIZE = 16
EPOCHS = 200
PATIENCE_ES = 60
LEARNING_RATE = 1e-4
L2_REG = 1e-4

AUGMENT_PROB = 0.6
NOISE_STD_RATIO = 0.02
SCALE_RANGE = (0.9, 1.1)
JITTER_MS = 50

N_CSP_COMPONENTS = 6
USE_ENSEMBLE = True

LOG_INTERVAL = 50
WANTED_CODES = {769, 770, 771, 772}
EOG_NAME_SUBSTRINGS = ['EOG', 'HEOG', 'VEOG']

# ---------- helpers ----------
def salvar_resultados_append(arq, texto):
    with open(arq, 'a', encoding='utf-8') as f:
        f.write(texto + "\n")

# ---------- load + preprocess ----------
def carregar_dados_gdf_preprocess(arquivo_path, apply_car=True):
    print(f"\nCarregando arquivo: {arquivo_path} ...")
    raw = mne.io.read_raw_gdf(arquivo_path, preload=True, verbose='ERROR')

    chs_to_drop = []
    for ch in raw.ch_names:
        for substr in EOG_NAME_SUBSTRINGS:
            if substr.lower() in ch.lower():
                chs_to_drop.append(ch)
                break
    if chs_to_drop:
        print(f"Removendo canais EOG detectados (evitar leakage): {chs_to_drop}")
        raw.drop_channels(chs_to_drop)

    raw.filter(F_MIN, F_MAX, fir_design='firwin', verbose='ERROR')

    events, event_id_map = mne.events_from_annotations(raw)
    print("event_id_map:", event_id_map)
    event_ids = {k: int(v) for k, v in event_id_map.items() if isinstance(v, (int, np.integer)) and int(v) in WANTED_CODES}
    if len(event_ids) == 0:
        for k, v in event_id_map.items():
            try:
                key_int = int(str(k))
                if key_int in WANTED_CODES:
                    event_ids[str(key_int)] = int(v)
            except Exception:
                pass
    unique_vals = np.unique(events[:, -1])
    if len(event_ids) == 0:
        chosen = np.sort(unique_vals)[:4]
        event_ids = {str(int(ch)): int(ch) for ch in chosen}
        print("Fallback event_ids:", event_ids)

    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
    if apply_car:
        try:
            raw.set_eeg_reference('average', projection=False)
            print("Applied average reference.")
        except Exception:
            pass

    epochs = mne.Epochs(raw, events, event_id=event_ids,
                        tmin=EPOCH_T_MIN, tmax=EPOCH_T_MAX,
                        proj=False, picks=picks,
                        baseline=(BASELINE_T_MIN, BASELINE_T_MAX),
                        preload=True, verbose='ERROR')
    epochs.crop(tmin=CROP_T_MIN, tmax=CROP_T_MAX)
    X = epochs.get_data()
    y_raw = epochs.events[:, -1]
    unique_vals = np.unique(y_raw)
    if set(WANTED_CODES).issubset(set(unique_vals)):
        base = min(WANTED_CODES)
        label_map = {val: (val - base) for val in unique_vals if int(val) in WANTED_CODES}
        y = np.vectorize(lambda v: label_map[int(v)])(y_raw)
    else:
        sorted_vals = np.sort(unique_vals)
        mapping = {val: idx for idx, val in enumerate(sorted_vals)}
        y = np.vectorize(mapping.get)(y_raw)

    print(f"Extraído: {X.shape[0]} trials, {X.shape[1]} canais (EEG), {X.shape[2]} amostras.")
    print("Canais (exemplo 10):", epochs.ch_names[:10])
    print("Labels únicos:", np.unique(y))
    return X, y

# ---------- augment ----------
def jitter_trial(trial, sfreq, jitter_ms=JITTER_MS):
    max_shift = int((jitter_ms / 1000.0) * sfreq)
    shift = GLOBAL_RNG.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return trial
    n_ch, n_samp = trial.shape
    out = np.zeros_like(trial)
    if shift > 0:
        out[:, :-shift] = trial[:, shift:]
    else:
        out[:, -shift:] = trial[:, :n_samp+shift]
    return out

def augment_batch(X_batch, sfreq):
    X_batch_aug = X_batch.copy()
    for i in range(X_batch.shape[0]):
        if GLOBAL_RNG.rand() < AUGMENT_PROB:
            tr = X_batch_aug[i]
            tr = jitter_trial(tr, sfreq)
            channel_stds = tr.std(axis=1, keepdims=True)
            noise = GLOBAL_RNG.normal(0, NOISE_STD_RATIO, size=tr.shape) * channel_stds
            tr = tr + noise
            scales = GLOBAL_RNG.uniform(SCALE_RANGE[0], SCALE_RANGE[1], size=(tr.shape[0], 1))
            tr = tr * scales
            X_batch_aug[i] = tr
    return X_batch_aug

# ---------- generator ----------
class EEGDataGenerator(Sequence):
    def __init__(self, X, y, batch_size=16, augment=False, shuffle=True, sfreq=250):
        self.X = X.astype(np.float32)
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.sfreq = sfreq
        self.indexes = np.arange(len(X))
        self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))
    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        Xb = self.X[batch_idx].copy()
        yb = self.y[batch_idx]
        if self.augment:
            Xb = augment_batch(Xb, self.sfreq)
        return Xb, yb
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = GLOBAL_RNG.permutation(self.indexes)

# ---------- model ----------
def criar_modelo_eegnet_vfinal(n_canais, n_amostras, n_classes,
                               F1=8, D=2, F2=16, dropout_dense=0.3, l2_reg=L2_REG):
    model = Sequential()
    model.add(InputLayer(input_shape=(n_canais, n_amostras)))
    model.add(Reshape((n_canais, n_amostras, 1)))
    model.add(Conv2D(F1, (1, 64), padding='same', use_bias=False,
                     kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization()); model.add(Activation('elu'))
    model.add(DepthwiseConv2D((n_canais, 1), use_bias=False, depth_multiplier=D))
    model.add(BatchNormalization()); model.add(Activation('elu'))
    model.add(AveragePooling2D((1, 4))); model.add(SpatialDropout2D(0.2))
    model.add(SeparableConv2D(F2, (1, 16), padding='same', use_bias=False))
    model.add(BatchNormalization()); model.add(Activation('elu'))
    model.add(AveragePooling2D((1, 8))); model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='elu', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(dropout_dense))
    model.add(Dense(n_classes, activation='softmax'))
    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# ---------- IntervalLogger ----------
class IntervalLogger(Callback):
    def __init__(self, interval=50):
        super().__init__()
        self.interval = max(1, int(interval))
        self.best_val_acc = -np.inf
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > self.best_val_acc:
            self.best_val_acc = logs.get('val_accuracy')
        if (epoch + 1) % self.interval == 0:
            self._print_summary(epoch + 1)
    def on_train_end(self, logs=None):
        self._print_summary('final')
    def _print_summary(self, epoch_label):
        last_train_acc = self.history['accuracy'][-1] if self.history['accuracy'] else None
        last_train_loss = self.history['loss'][-1] if self.history['loss'] else None
        last_val_acc = self.history['val_accuracy'][-1] if self.history['val_accuracy'] else None
        last_val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else None
        lr = None
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        except Exception:
            pass
        print("\n" + "="*40)
        print(f"Resumo - época: {epoch_label}")
        if last_train_acc is not None:
            print(f"  train acc: {last_train_acc:.4f} | train loss: {last_train_loss:.4f}")
        if last_val_acc is not None:
            print(f"  val acc:   {last_val_acc:.4f} | val loss:   {last_val_loss:.4f}")
        if lr is not None:
            print(f"  lr: {lr:.6g}")
        print(f"  best val_acc seen: {self.best_val_acc:.4f}")
        print("="*40 + "\n")

# ---------- bootstrap helper ----------
def bootstrap_ci(data, n_boot=2000, alpha=0.05):
    rng = GLOBAL_RNG
    boot_stats = []
    n = len(data)
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats.append(np.mean(sample))
    lower = np.percentile(boot_stats, 100 * (alpha/2))
    upper = np.percentile(boot_stats, 100 * (1 - alpha/2))
    return lower, upper

# ---------- main ----------
def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    early_stopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=PATIENCE_ES,
                                  verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1, min_lr=1e-6)

    arquivos = sorted(glob.glob(os.path.join(DATA_PATH, 'A0?T.gdf')))
    if not arquivos:
        print("Nenhum arquivo A0?T.gdf encontrado em", DATA_PATH)
        return

    all_accuracies = []

    for arquivo in arquivos:
        subject_label = os.path.basename(arquivo)[:3]
        print("\n" + "#"*50)
        print(f"Processando {subject_label} ({arquivo})")
        print("#"*50)

        try:
            X_sujeito, y_sujeito = carregar_dados_gdf_preprocess(arquivo, apply_car=True)
        except Exception as e:
            print("Erro ao carregar:", e)
            salvar_resultados_append(OUTPUT_FILE, f"{subject_label} - erro ao carregar: {e}")
            continue

        if len(np.unique(y_sujeito)) < N_CLASSES:
            msg = f"Subject {subject_label} tem menos de {N_CLASSES} classes. Pulando."
            print(msg); salvar_resultados_append(OUTPUT_FILE, msg); continue

        y_cat = to_categorical(y_sujeito, num_classes=N_CLASSES)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_idx = 1
        subject_accs = []

        for train_idx, test_idx in skf.split(X_sujeito, y_sujeito):
            print(f"\n--- {subject_label} Fold {fold_idx}/{N_SPLITS} ---")
            X_train_raw, X_test_raw = X_sujeito[train_idx], X_sujeito[test_idx]
            y_train_cat, y_test_cat = y_cat[train_idx], y_cat[test_idx]
            y_train_labels, y_test_labels = y_sujeito[train_idx], y_sujeito[test_idx]

            mean = X_train_raw.mean(axis=(0, 2), keepdims=True)
            std = X_train_raw.std(axis=(0, 2), keepdims=True) + 1e-10
            X_train = (X_train_raw - mean) / std
            X_test = (X_test_raw - mean) / std

            lda = None; lda_proba_test = None; lda_acc = None
            try:
                # REMOVED random_state argument
                csp = CSP(n_components=N_CSP_COMPONENTS, reg=None, log=None, transform_into='average_power')
                csp.fit(X_train, y_train_labels)
                X_train_csp = csp.transform(X_train)
                X_test_csp = csp.transform(X_test)
                if X_train_csp.ndim != 2 or X_test_csp.ndim != 2:
                    raise ValueError("CSP outputs not 2D as expected.")
                scaler_csp = StandardScaler().fit(X_train_csp)
                X_train_csp = scaler_csp.transform(X_train_csp)
                X_test_csp = scaler_csp.transform(X_test_csp)
                lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                lda.fit(X_train_csp, y_train_labels)
                lda_proba_test = lda.predict_proba(X_test_csp)
                lda_acc = accuracy_score(y_test_labels, lda.predict(X_test_csp))
            except Exception as e:
                print("Erro CSP/LDA:", e)
                lda_proba_test = None; lda_acc = None

            n_trials, n_channels, n_samples = X_train.shape
            model = criar_modelo_eegnet_vfinal(n_channels, n_samples, N_CLASSES, dropout_dense=0.3)

            classes = np.unique(y_train_labels)
            class_weights_arr = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_labels)
            class_weight_dict = {int(cls): w for cls, w in zip(classes, class_weights_arr)}

            train_gen = EEGDataGenerator(X_train, y_train_cat, batch_size=BATCH_SIZE, augment=True, shuffle=True, sfreq=250)
            val_gen = EEGDataGenerator(X_test, y_test_cat, batch_size=BATCH_SIZE, augment=False, shuffle=False, sfreq=250)

            checkpoint_path = f'best_{subject_label}_fold{fold_idx}.keras'
            try:
                checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', mode='max',
                                             save_best_only=True, verbose=0, save_format='keras')
            except TypeError:
                checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', mode='max',
                                             save_best_only=True, verbose=0)

            interval_logger = IntervalLogger(interval=LOG_INTERVAL)

            # NOTE: removed workers/use_multiprocessing args for compatibility
            history = model.fit(train_gen,
                                validation_data=val_gen,
                                epochs=EPOCHS,
                                callbacks=[early_stopper, reduce_lr, checkpoint, interval_logger],
                                verbose=0,
                                class_weight=class_weight_dict)

            if os.path.exists(checkpoint_path):
                try:
                    model.load_weights(checkpoint_path)
                except Exception:
                    pass

            y_pred_proba_eeg = model.predict(X_test, batch_size=BATCH_SIZE)
            y_pred_eeg = np.argmax(y_pred_proba_eeg, axis=1)
            eeg_acc = accuracy_score(y_test_labels, y_pred_eeg)

            if USE_ENSEMBLE and lda_proba_test is not None:
                lda_order = list(lda.classes_)
                if list(lda_order) != list(range(N_CLASSES)):
                    reordered = np.zeros((lda_proba_test.shape[0], N_CLASSES))
                    for col_idx, cls in enumerate(lda_order):
                        reordered[:, int(cls)] = lda_proba_test[:, col_idx]
                    lda_proba_reordered = reordered
                else:
                    lda_proba_reordered = lda_proba_test
                ensemble_proba = (y_pred_proba_eeg + lda_proba_reordered) / 2.0
                y_pred_ens = np.argmax(ensemble_proba, axis=1)
                ens_acc = accuracy_score(y_test_labels, y_pred_ens)
            else:
                ens_acc = None

            msg = f"{subject_label} Fold {fold_idx}: EEGNet acc={eeg_acc*100:.2f}%"
            if lda_acc is not None:
                msg += f" | LDA(CSP) acc={lda_acc*100:.2f}%"
            if ens_acc is not None:
                msg += f" | Ensemble acc={ens_acc*100:.2f}%"
            print(msg); salvar_resultados_append(OUTPUT_FILE, msg)

            salvar_resultados_append(OUTPUT_FILE, f"{subject_label} Fold {fold_idx} - EEGNet report:")
            salvar_resultados_append(OUTPUT_FILE, classification_report(y_test_labels, y_pred_eeg, target_names=CLASSES, zero_division=0))
            if lda_acc is not None:
                salvar_resultados_append(OUTPUT_FILE, f"{subject_label} Fold {fold_idx} - LDA report:")
                salvar_resultados_append(OUTPUT_FILE, classification_report(y_test_labels, lda.predict(X_test_csp), target_names=CLASSES, zero_division=0))
            if ens_acc is not None:
                salvar_resultados_append(OUTPUT_FILE, f"{subject_label} Fold {fold_idx} - Ensemble report:")
                salvar_resultados_append(OUTPUT_FILE, classification_report(y_test_labels, y_pred_ens, target_names=CLASSES, zero_division=0))

            cm = confusion_matrix(y_test_labels, y_pred_eeg, labels=list(range(N_CLASSES)))
            salvar_resultados_append(OUTPUT_FILE, f"Confusion EEGNet fold{fold_idx}:\n{cm}")
            if lda_acc is not None:
                cm_lda = confusion_matrix(y_test_labels, lda.predict(X_test_csp), labels=list(range(N_CLASSES)))
                salvar_resultados_append(OUTPUT_FILE, f"Confusion LDA fold{fold_idx}:\n{cm_lda}")
            if ens_acc is not None:
                cm_ens = confusion_matrix(y_test_labels, y_pred_ens, labels=list(range(N_CLASSES)))
                salvar_resultados_append(OUTPUT_FILE, f"Confusion ENS fold{fold_idx}:\n{cm_ens}")

            used_acc = ens_acc if (ens_acc is not None) else eeg_acc
            subject_accs.append(used_acc)
            all_accuracies.append(used_acc if used_acc is not None else eeg_acc)

            fold_idx += 1

        if subject_accs:
            mean_sub = np.mean(subject_accs) * 100
            msg_sub = f"{subject_label} mean acc = {mean_sub:.2f}%"
            print(msg_sub); salvar_resultados_append(OUTPUT_FILE, msg_sub)

    if all_accuracies:
        overall_mean = np.mean(all_accuracies) * 100
        lower, upper = bootstrap_ci(np.array(all_accuracies))
        final_msg = f"\n=== Acurácia média FINAL: {overall_mean:.2f}% | 95% CI (boot) = [{lower*100:.2f}%, {upper*100:.2f}%] ==="
        print(final_msg); salvar_resultados_append(OUTPUT_FILE, final_msg)
    else:
        print("Nenhuma acurácia coletada.")

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTempo total: {time.time() - start:.1f}s")
