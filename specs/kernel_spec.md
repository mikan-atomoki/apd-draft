# APD Intelligibility Estimator - カスタムカーネル仕様書

## 概要

APD（聴覚情報処理障害）当事者にとっての音声了解度をリアルタイム推定するモデルの、Android向けカスタム推論カーネル仕様。

BitNet（1-bit重み）を活かすため、ONNX等の汎用ランタイムではなくカスタムカーネルで推論を行う。

**重要: このモデルはBitNet層とFP32層が混在する。**
- FP32層: エンコーダ第1層 (生波形入力)、Depthwise Conv (重み数が少なすぎて量子化不可)
- BitNet層: Bottleneck、TCN Input、Pointwise Conv、Head

---

## モデル仕様

| 項目 | 値 |
|------|-----|
| 入力 | 16kHz mono audio, 1秒窓 (shape: `[1, 1, 16000]`) |
| 出力 | 了解度スコア `float32` (0.0 - 1.0) |
| 重みフォーマット | BitNet層: 1-bit packed (`int8` に 8 weights pack) / FP32層: `float32` |
| スケールファクタ | BitNet層ごとに `float32` × 1 |
| バイアス | 一部層のみ `float32` |
| 目標推論時間 | < 10ms / 1秒窓 (Snapdragon 7xx 以上) |
| 目標モデルサイズ | < 2MB (プルーニング + BitNet後) |

---

## 重みフォーマット (`.apd` カスタムフォーマット)

### ファイル構造

```
[Header]
  magic: "APD1" (4 bytes)
  version: uint16
  n_layers: uint16
  sample_rate: uint32 (= 16000)
  window_size: uint32 (= 16000)

[Layer Table] × n_layers
  layer_type: uint8
    0 = BitConv1d       (1-bit weights, absmax正規化あり)
    1 = BitLinear       (1-bit weights, absmax正規化あり)
    2 = FP32Conv1d      (通常のconv1d, エンコーダ第1層・depthwise用)
    3 = FP32Linear      (通常のlinear, fallback用)
    4 = GroupNorm
    5 = PReLU
  name_len: uint16
  name: char[name_len]

  # type 0: BitConv1d
  in_channels: uint16
  out_channels: uint16
  kernel_size: uint16
  stride: uint16
  padding: uint16
  dilation: uint16
  groups: uint16
  weight_offset: uint64    # packed 1-bit weights の開始位置
  weight_size: uint64      # bytes
  scale: float32
  has_bias: uint8
  bias_offset: uint64      # float32[] の開始位置

  # type 1: BitLinear
  in_features: uint32
  out_features: uint32
  weight_offset: uint64
  weight_size: uint64
  scale: float32
  has_bias: uint8
  bias_offset: uint64

  # type 2: FP32Conv1d
  in_channels: uint16
  out_channels: uint16
  kernel_size: uint16
  stride: uint16
  padding: uint16
  dilation: uint16
  groups: uint16
  weight_offset: uint64    # float32[] weights
  weight_size: uint64
  has_bias: uint8
  bias_offset: uint64

  # type 3: FP32Linear
  in_features: uint32
  out_features: uint32
  weight_offset: uint64    # float32[] weights
  has_bias: uint8
  bias_offset: uint64

  # type 4: GroupNorm
  num_groups: uint16
  num_channels: uint16
  weight_offset: uint64    # float32 gamma
  bias_offset: uint64      # float32 beta
  eps: float32

  # type 5: PReLU
  num_parameters: uint16
  weight_offset: uint64    # float32[]

[Weight Data]
  # 全層の重みデータが連続して格納
  # BitNet重み: 1-bit packed into int8 (8 weights per byte, MSB first)
  # FP32重み/バイアス: little-endian float32
```

### 1-bit weight packing

```
Packing rule:
  sign(w) == +1  → bit = 1
  sign(w) == -1  → bit = 0

  8 weights → 1 byte (MSB first)

  Example: weights [+1, -1, +1, +1, -1, -1, +1, -1]
           bits    [ 1,  0,  1,  1,  0,  0,  1,  0]
           byte    = 0b10110010 = 0xB2

Padding: 末尾が8の倍数でない場合、0でパディング
```

---

## カーネル実装仕様

### 1. FP32Conv1d カーネル (エンコーダ第1層・Depthwise Conv用)

```
入力:  x[batch][in_ch][time]     (float32)
重み:  w[out_ch][in_ch/groups][kernel_size]  (float32)
出力:  y[batch][out_ch][out_time] (float32)

手順:
  標準の畳み込み演算 (dilation, groups 対応)
  out_time = (time + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
```

**最適化ヒント:**
- Depthwise (groups == in_channels): チャネルごとに独立、NEON並列化が容易
- エンコーダ (stride=20): 出力フレーム数が大幅に減るため比較的軽量

### 2. BitConv1d カーネル (Pointwise Conv・Bottleneck用)

```
入力:  x[batch][in_ch][time]     (float32)
重み:  w_packed[out_ch][in_ch/groups][kernel_size/8]  (int8, packed bits)
出力:  y[batch][out_ch][out_time] (float32)

手順:
  1. x を absmax 正規化: x_norm = x / mean(|x|)
  2. packed 1-bit重みを展開しながら accumulate:
     - bit=1 → +1, bit=0 → -1
     - 乗算不要: bit=1 なら加算、bit=0 なら減算
  3. y = accumulator * w_scale * x_scale * layer_scale
```

**最適化ヒント:**
- Pointwise (kernel_size=1) が大半なので、実質行列積 → popcount最適化が効く
- NEON SIMD で 128-bit レジスタに 128 weights を一度にロード
- `vcnt` (popcount) で +1 の数をカウント → `count*2 - total` で sum 計算

### 3. BitLinear カーネル

```
入力:  x[batch][in_features]    (float32)
重み:  w_packed[out_features][in_features/8]  (int8, packed bits)
出力:  y[batch][out_features]   (float32)

手順:
  1. x を absmax 正規化
  2. popcount ベースの内積計算:
     - x を符号ビットに変換
     - XNOR(x_bits, w_bits) → popcount → sum
  3. y = sum * w_scale * x_scale * layer_scale + bias
```

### 3. GroupNorm

```
標準実装でOK (float32)
channels を num_groups に分割、各グループ内で正規化
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

### 4. PReLU

```
y = x if x >= 0 else alpha * x
alpha は学習済みパラメータ (float32, per-channel)
```

### 5. Global Average Pooling

```
y[batch][ch] = mean(x[batch][ch][:])  over time axis
```

### 6. Sigmoid

```
y = 1 / (1 + exp(-x))
近似可: fast sigmoid lookup table (256 entries, 精度十分)
```

---

## 推論パイプライン (実行順序)

```
Input: float32[1][1][16000]
  │
  ├─ FP32Conv1d (encoder)    → [1][512][F]     F = (16000+40)/20 程度
  ├─ GroupNorm
  ├─ PReLU
  │
  ├─ BitConv1d (bottleneck)  → [1][256][F]
  ├─ BitConv1d (tcn_input)   → [1][512][F]
  │
  ├─ TCN Block ×3
  │   └─ DepthwiseSeparableConv ×8 (per block, dilation=2^i)
  │       ├─ FP32Conv1d (depthwise, groups=512, dilation=2^i)
  │       ├─ GroupNorm
  │       ├─ PReLU
  │       ├─ BitConv1d (pointwise, kernel=1)
  │       └─ Residual add
  │
  ├─ Global Average Pooling  → [1][512]
  ├─ BitLinear               → [1][256]
  ├─ PReLU
  ├─ Dropout (推論時skip)
  ├─ BitLinear (or FP32)     → [1][1]
  └─ Sigmoid                 → [1][1]  ← 了解度スコア
```

---

## Android 統合仕様

### Java/Kotlin インターフェース

```kotlin
class APDInference(context: Context) {

    /**
     * モデルファイル (.apd) をロード
     * assets/ に配置する想定
     */
    fun loadModel(assetPath: String): Boolean

    /**
     * 1秒分の音声からスコアを推定
     * @param audio PCM float32, 16kHz mono, length=16000
     * @return 了解度スコア 0.0-1.0 (0=聴き取り不可, 1=問題なし)
     */
    fun estimateIntelligibility(audio: FloatArray): Float

    /**
     * リソース解放
     */
    fun release()
}
```

### JNI ブリッジ

```
native fun nativeLoadModel(assetManager: AssetManager, path: String): Long
native fun nativeInfer(handle: Long, audio: FloatArray): Float
native fun nativeRelease(handle: Long)
```

### NDK (C++) 側

```cpp
// apd_inference.h
class APDInference {
public:
    bool loadModel(const uint8_t* data, size_t size);
    float infer(const float* audio_16k_mono, size_t length);  // length=16000
    ~APDInference();

private:
    std::vector<Layer> layers_;
    // 中間バッファは事前確保してreuse (GC回避)
    std::vector<float> buffer_a_;
    std::vector<float> buffer_b_;
};
```

---

## 閾値設定 (フロントエンド向け参考値)

| スコア | 意味 | 推奨アクション |
|--------|------|----------------|
| 0.8 - 1.0 | 問題なし | 表示のみ (緑) |
| 0.5 - 0.8 | やや困難 | 注意表示 (黄) |
| 0.3 - 0.5 | 困難 | 警告 (橙) |
| 0.0 - 0.3 | 聴き取り不可能 | 強い警告 (赤) |

※ 閾値は当事者のフィードバックを元にキャリブレーション想定

---

## エクスポートツール (モデル開発者 → アプリ開発者)

モデル開発者（AI側）が提供するスクリプト:

```bash
python export_apd.py \
    --checkpoint model_pruned_bitnet.pt \
    --output model.apd \
    --validate  # テスト入力で PyTorch と .apd 推論結果を比較
```

このスクリプトは AI 側 (モデル開発者) が実装・提供する。
