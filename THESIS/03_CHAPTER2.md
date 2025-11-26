# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1. Công nghệ viễn thám và ảnh vệ tinh

### 2.1.1. Nguyên lý viễn thám

Viễn thám (Remote Sensing) là khoa học và kỹ thuật thu thập thông tin về một đối tượng hoặc khu vực từ xa, thường thông qua việc ghi nhận bức xạ điện từ phản xạ hoặc phát ra từ bề mặt Trái Đất. Nguyên lý cơ bản của viễn thám dựa trên tương tác giữa bức xạ điện từ và các đối tượng trên bề mặt:

**Quá trình viễn thám bị động (Passive Remote Sensing):**

1. **Nguồn năng lượng:** Mặt Trời phát ra bức xạ điện từ
2. **Truyền qua khí quyển:** Một phần bức xạ bị hấp thụ và tán xạ bởi khí quyển
3. **Tương tác với bề mặt:** Bức xạ phản xạ, hấp thụ, và truyền qua tùy theo đặc tính vật liệu
4. **Ghi nhận bởi cảm biến:** Vệ tinh thu nhận bức xạ phản xạ
5. **Truyền dữ liệu:** Tín hiệu được truyền về trạm mặt đất

**Phương trình cân bằng năng lượng:**

```
E_incident = E_reflected + E_absorbed + E_transmitted
```

Trong đó:
- E_incident: Năng lượng tới (từ Mặt Trời)
- E_reflected: Năng lượng phản xạ (được cảm biến ghi nhận)
- E_absorbed: Năng lượng hấp thụ (chuyển thành nhiệt)
- E_transmitted: Năng lượng truyền qua

**Hệ số phản xạ phổ (Spectral Reflectance):**

```
ρ(λ) = E_reflected(λ) / E_incident(λ)
```

Trong đó:
- ρ(λ): Hệ số phản xạ tại bước sóng λ
- Giá trị từ 0 (hấp thụ hoàn toàn) đến 1 (phản xạ hoàn toàn)

### 2.1.2. Radar khẩu độ tổng hợp (SAR)

**Nguyên lý hoạt động:**

Khác với viễn thám bị động, SAR là hệ thống chủ động (active remote sensing):

1. **Phát xung radar:** Anten phát xung sóng điện từ về phía Trái Đất
2. **Tương tác với bề mặt:** Sóng radar phản xạ ngược (backscatter) với cường độ phụ thuộc vào:
   - Độ nhám bề mặt (surface roughness)
   - Độ ẩm (moisture content)
   - Hằng số điện môi (dielectric constant)
   - Góc tới (incidence angle)
3. **Thu tín hiệu phản xạ:** Anten thu nhận tín hiệu backscatter
4. **Xử lý tín hiệu:** Tổng hợp khẩu độ để tăng độ phân giải

**Phương trình Radar:**

```
P_r = (P_t × G² × λ² × σ⁰) / ((4π)³ × R⁴)
```

Trong đó:
- P_r: Công suất nhận
- P_t: Công suất phát
- G: Độ lợi anten
- λ: Bước sóng
- σ⁰: Hệ số backscatter (radar cross-section per unit area)
- R: Khoảng cách từ radar đến bề mặt

**Hệ số Backscatter (σ⁰):**

```
σ⁰ (dB) = 10 × log₁₀(σ⁰_linear)
```

Giá trị σ⁰ phụ thuộc vào:
- **Độ nhám bề mặt:** Bề mặt nhẵn (nước) → σ⁰ thấp, bề mặt nhám (rừng) → σ⁰ cao
- **Độ ẩm:** Độ ẩm cao → σ⁰ cao (nước có hằng số điện môi lớn)
- **Cấu trúc thực vật:** Rừng có cấu trúc phức tạp → backscatter mạnh

**Polarization:**

SAR có thể phát và thu theo các polarization khác nhau:
- **VV:** Phát V (Vertical), Thu V → Nhạy với độ ẩm bề mặt
- **VH:** Phát V, Thu H (Horizontal) → Nhạy với cấu trúc thực vật (volume scattering)
- **HH:** Phát H, Thu H → Nhạy với độ nhám bề mặt
- **HV:** Phát H, Thu V → Tương tự VH

**Sentinel-1 SAR:**
- Dải sóng: C-band (λ = 5.5 cm, frequency = 5.4 GHz)
- Polarization: VV và VH (IW mode)
- Độ phân giải không gian: 10m
- Ưu điểm: Xuyên qua mây, hoạt động ngày/đêm

### 2.1.3. Ảnh quang học đa phổ (Optical Multispectral)

**Dải phổ điện từ:**

Ảnh quang học ghi nhận bức xạ phản xạ từ bề mặt Trái Đất ở các dải phổ khác nhau:

1. **Visible (VIS):** 400-700 nm
   - Blue (B): 450-520 nm
   - Green (G): 520-600 nm
   - Red (R): 630-690 nm

2. **Near-Infrared (NIR):** 700-1400 nm
   - Phản xạ cao ở thực vật xanh (chlorophyll)
   - Quan trọng cho tính toán NDVI

3. **Short-Wave Infrared (SWIR):** 1400-3000 nm
   - SWIR1: 1550-1750 nm
   - SWIR2: 2080-2350 nm
   - Nhạy với độ ẩm thực vật và đất

**Chữ ký phổ (Spectral Signature):**

Mỗi loại đối tượng có chữ ký phổ đặc trưng - mẫu phản xạ qua các dải phổ:

```
S = [ρ(λ₁), ρ(λ₂), ..., ρ(λₙ)]
```

Ví dụ:
- **Thực vật xanh:** Phản xạ thấp ở Red (hấp thụ bởi chlorophyll), phản xạ cao ở NIR
- **Đất trống:** Phản xạ trung bình và tăng dần theo bước sóng
- **Nước:** Phản xạ thấp ở tất cả các dải (đặc biệt NIR và SWIR)

**Sentinel-2 Multispectral Imager:**

| Band | Tên | Bước sóng (nm) | Độ phân giải (m) | Ứng dụng |
|------|-----|---------------|------------------|----------|
| B2 | Blue | 490 | 10 | Phân biệt đất/nước |
| B3 | Green | 560 | 10 | Đánh giá thực vật |
| B4 | Red | 665 | 10 | Chlorophyll absorption |
| B8 | NIR | 842 | 10 | Biomass, NDVI |
| B11 | SWIR1 | 1610 | 20 | Độ ẩm, NDMI |
| B12 | SWIR2 | 2190 | 20 | Phân biệt đất/rừng, NBR |

### 2.1.4. Chỉ số thực vật

**NDVI (Normalized Difference Vegetation Index):**

```
NDVI = (NIR - Red) / (NIR + Red)
```

*Lưu ý: Trong thực tế, một epsilon nhỏ (1e-8) được thêm vào mẫu số để tránh chia cho 0.*

**Nguyên lý:**
- Thực vật xanh: Hấp thụ mạnh Red (chlorophyll), phản xạ cao NIR (cấu trúc tế bào) → NDVI cao
- Đất trống/nước: Phản xạ thấp cả Red và NIR → NDVI thấp

**Phạm vi giá trị:**
- NDVI > 0.6: Thực vật xanh tốt (rừng rậm)
- 0.2 < NDVI < 0.6: Thực vật thưa, cỏ
- NDVI < 0.2: Đất trống, nước, đô thị

**Đạo hàm toán học:**
```
∂NDVI/∂NIR = (2 × Red) / (NIR + Red)²
∂NDVI/∂Red = -(2 × NIR) / (NIR + Red)²
```

**NBR (Normalized Burn Ratio):**

```
NBR = (NIR - SWIR2) / (NIR + SWIR2)
```

*Lưu ý: Trong triển khai, epsilon (1e-8) được thêm vào mẫu số.*

**Nguyên lý:**
- NIR: Phản xạ cao ở thực vật xanh
- SWIR2: Nhạy với độ ẩm và vùng cháy
- Vùng cháy: NIR giảm, SWIR2 tăng → NBR giảm mạnh

**Delta NBR (dNBR):**
```
dNBR = NBR_before - NBR_after
```

- dNBR > 0.66: Cháy nghiêm trọng
- 0.44 < dNBR < 0.66: Cháy vừa
- 0.27 < dNBR < 0.44: Cháy nhẹ
- dNBR < 0.27: Không cháy hoặc tái sinh

**NDMI (Normalized Difference Moisture Index):**

```
NDMI = (NIR - SWIR1) / (NIR + SWIR1)
```

*Lưu ý: Trong triển khai, epsilon (1e-8) được thêm vào mẫu số.*

**Nguyên lý:**
- SWIR1 (~1600 nm): Hấp thụ mạnh bởi nước
- Độ ẩm thực vật cao → SWIR1 phản xạ thấp → NDMI cao
- Stress hạn → NDMI giảm

**Phạm vi giá trị:**
- NDMI > 0.4: Độ ẩm cao
- 0.0 < NDMI < 0.4: Độ ẩm trung bình
- NDMI < 0: Stress hạn, nguy cơ cháy cao

### 2.1.5. Phát hiện biến động rừng

**Change Detection Approach:**

```
ΔFeature = Feature_after - Feature_before
```

**Temporal Features:**
- **Before features:** Trạng thái rừng tại thời điểm t₁
- **After features:** Trạng thái rừng tại thời điểm t₂
- **Delta features:** Biến đổi giữa hai thời điểm (t₂ - t₁)

**Ví dụ với NDVI:**

```
ΔNDVI = NDVI_after - NDVI_before
```

**Phân loại biến động:**
- ΔNDVI << 0 (giảm mạnh): Mất rừng (deforestation)
- ΔNDVI ≈ 0: Rừng ổn định
- ΔNDVI >> 0 (tăng mạnh): Tái trồng rừng

**Kết hợp đa chỉ số:**

```
Change_score = w₁×|ΔNDVI| + w₂×|ΔNBR| + w₃×|ΔNDMI|
```

Trong đó w₁, w₂, w₃ là trọng số được học từ dữ liệu.

---

## 2.2. Mạng Neural Tích chập (Convolutional Neural Networks)

### 2.2.1. Giới thiệu về Neural Networks

**Perceptron - Đơn vị cơ bản:**

Một neuron nhân tạo thực hiện phép biến đổi tuyến tính và hàm kích hoạt:

```
y = f(w^T × x + b)
```

Trong đó:
- x ∈ ℝⁿ: Input vector (n features)
- w ∈ ℝⁿ: Weight vector
- b ∈ ℝ: Bias
- f(.): Activation function
- y: Output

**Multi-Layer Perceptron (MLP):**

Một mạng neural gồm nhiều layers:

```
Layer 1: h₁ = f₁(W₁×x + b₁)
Layer 2: h₂ = f₂(W₂×h₁ + b₂)
...
Output: y = fₙ(Wₙ×hₙ₋₁ + bₙ)
```

**Universal Approximation Theorem:**

Một MLP với ít nhất một hidden layer và đủ neurons có thể xấp xỉ bất kỳ hàm liên tục nào với độ chính xác tùy ý trên một compact subset.

### 2.2.2. Convolutional Layer

**Phép tích chập 2D (2D Convolution):**

Đây là thành phần cốt lõi của CNN, thực hiện phép tích chập giữa input và kernel:

```
(I * K)(i, j) = ΣΣ I(i+m, j+n) × K(m, n)
               m n
```

Trong đó:
- I: Input feature map (height × width × channels)
- K: Kernel/Filter (k_h × k_w × channels)
- (i, j): Vị trí output
- (m, n): Vị trí trong kernel

**Ví dụ cụ thể với kernel 3×3:**

Input I (5×5):
```
[1  2  3  4  5]
[6  7  8  9  10]
[11 12 13 14 15]
[16 17 18 19 20]
[21 22 23 24 25]
```

Kernel K (3×3):
```
[1  0 -1]
[1  0 -1]
[1  0 -1]
```

Output tại (1,1):
```
O(1,1) = 1×1 + 2×0 + 3×(-1) +
         6×1 + 7×0 + 8×(-1) +
         11×1 + 12×0 + 13×(-1)
       = 1 + 0 - 3 + 6 + 0 - 8 + 11 + 0 - 13
       = -6
```

**Công thức tổng quát cho Multi-Channel Convolution:**

```
O(i, j, k) = Σ   Σ  Σ  I(i+m, j+n, c) × K(m, n, c, k) + b_k
             c   m  n
```

Trong đó:
- I ∈ ℝ^(H × W × C_in): Input với C_in channels
- K ∈ ℝ^(k_h × k_w × C_in × C_out): Kernel
- O ∈ ℝ^(H' × W' × C_out): Output với C_out channels
- b_k: Bias cho output channel k

**Output size calculation:**

```
H_out = ⌊(H_in + 2×padding - kernel_size) / stride⌋ + 1
W_out = ⌊(W_in + 2×padding - kernel_size) / stride⌋ + 1
```

**Parameters trong Convolutional Layer:**

```
#params = (k_h × k_w × C_in × C_out) + C_out
```

Trong đó:
- k_h × k_w × C_in × C_out: Weights
- C_out: Biases

**Ví dụ:** Conv2D(in_channels=27, out_channels=64, kernel_size=3)
```
#params = (3 × 3 × 27 × 64) + 64
        = 15,552 + 64
        = 15,616
```

**Ưu điểm của Convolution:**

1. **Parameter sharing:** Cùng một kernel được áp dụng cho toàn bộ input
   - MLP: Mỗi connection có weight riêng → O(H×W) parameters
   - CNN: Kernel được chia sẻ → O(k²) parameters (k << H, W)

2. **Translation invariance:** Nhận diện đặc trưng ở bất kỳ vị trí nào
   - Nếu đối tượng dịch chuyển, CNN vẫn nhận diện được

3. **Local connectivity:** Mỗi neuron chỉ kết nối với vùng local của input
   - Phù hợp với cấu trúc không gian của ảnh

### 2.2.3. Activation Functions

**ReLU (Rectified Linear Unit):**

```
f(x) = max(0, x) = {x  if x > 0
                    {0  if x ≤ 0
```

**Đạo hàm:**
```
f'(x) = {1  if x > 0
        {0  if x ≤ 0
```

**Ưu điểm:**
- Tính toán nhanh (không có exp, log)
- Giải quyết vanishing gradient problem
- Sparse activation (nhiều neurons = 0)
- Gradient đơn giản (0 hoặc 1)

**Nhược điểm:**
- Dying ReLU: Neurons có thể "chết" (output luôn 0) nếu gradient liên tục âm

**Leaky ReLU:**

```
f(x) = {x      if x > 0
       {α×x    if x ≤ 0
```

Với α = 0.01 (small positive slope cho x < 0)

**Softmax (cho Output Layer):**

```
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```

**Tính chất:**
- Output là xác suất: 0 ≤ softmax(x)ᵢ ≤ 1
- Tổng = 1: Σᵢ softmax(x)ᵢ = 1
- Dùng cho multi-class classification

**Ví dụ:**
```
Input logits: [2.0, 1.0, 0.1]

exp([2.0, 1.0, 0.1]) = [7.39, 2.72, 1.11]
Sum = 11.22

Softmax = [7.39/11.22, 2.72/11.22, 1.11/11.22]
        = [0.659, 0.242, 0.099]
```

### 2.2.4. Pooling Layers

**Max Pooling:**

```
MaxPool(i, j) = max  I(i+m, j+n)
                m,n∈W
```

Trong đó W là pooling window (thường 2×2)

**Ví dụ Max Pooling 2×2:**

Input (4×4):
```
[1  3  2  4]
[5  6  7  8]
[9  10 11 12]
[13 14 15 16]
```

Output (2×2):
```
[max(1,3,5,6)=6    max(2,4,7,8)=8   ]
[max(9,10,13,14)=14 max(11,12,15,16)=16]
```

Result:
```
[6  8 ]
[14 16]
```

**Average Pooling:**

```
AvgPool(i, j) = (1/|W|) × Σ I(i+m, j+n)
                         m,n∈W
```

**Global Average Pooling:**

```
GAP(k) = (1/(H×W)) × ΣΣ I(i, j, k)
                      i j
```

- Giảm spatial dimensions về 1×1
- Output: (1, 1, C) từ (H, W, C)
- Ưu điểm: Không có parameters, giảm overfitting

**Tác dụng của Pooling:**

1. **Downsampling:** Giảm spatial dimensions → giảm computation
2. **Translation invariance:** Tăng khả năng bất biến với phép dịch nhỏ
3. **Receptive field:** Tăng receptive field của các layers sau

### 2.2.5. Batch Normalization

**Motivation:**

Internal Covariate Shift: Phân bố input của mỗi layer thay đổi trong quá trình training, làm chậm quá trình hội tụ.

**Batch Normalization Algorithm:**

Với một mini-batch B = {x₁, x₂, ..., x_m}:

**Step 1: Tính mean và variance của batch**
```
μ_B = (1/m) × Σᵢ xᵢ

σ²_B = (1/m) × Σᵢ (xᵢ - μ_B)²
```

**Step 2: Normalize**
```
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
```

Trong đó ε = 1e-8 (tránh chia cho 0)

**Step 3: Scale and shift (learnable parameters)**
```
yᵢ = γ × x̂ᵢ + β
```

Trong đó:
- γ: Scale parameter (learnable)
- β: Shift parameter (learnable)

**Tại sao cần γ và β?**

Normalization có thể hạn chế khả năng biểu diễn của network. γ và β cho phép network học lại phân bố tối ưu.

**Inference (Test time):**

Sử dụng running mean và variance từ training:

```
μ_running = momentum × μ_running + (1-momentum) × μ_B
σ²_running = momentum × σ²_running + (1-momentum) × σ²_B

y_test = γ × (x - μ_running) / √(σ²_running + ε) + β
```

**Ưu điểm:**
- Tăng tốc training (có thể dùng learning rate cao hơn)
- Giảm sensitivity với weight initialization
- Có tác dụng regularization (tương tự dropout)
- Ổn định gradient flow

**Gradient Computation:**

```
∂L/∂x̂ᵢ = ∂L/∂yᵢ × γ

∂L/∂σ²_B = Σᵢ (∂L/∂x̂ᵢ × (xᵢ - μ_B)) × (-1/2) × (σ²_B + ε)^(-3/2)

∂L/∂μ_B = Σᵢ (∂L/∂x̂ᵢ × (-1/√(σ²_B + ε)))

∂L/∂xᵢ = ∂L/∂x̂ᵢ × (1/√(σ²_B + ε)) + (∂L/∂σ²_B × 2(xᵢ-μ_B)/m) + (∂L/∂μ_B/m)
```

### 2.2.6. Dropout

**Motivation:**

Overfitting xảy ra khi model học quá chi tiết training data, không generalize tốt cho unseen data.

**Dropout Algorithm (Training):**

```
For each neuron i:
    r_i ~ Bernoulli(p)  # Với xác suất p giữ lại neuron
    ỹ_i = r_i × y_i     # Nếu r_i=0, neuron bị tắt
```

Trong đó:
- p: Dropout probability (thường p=0.5 cho FC, p=0.7-0.9 cho Conv)
- r_i: Binary mask (0 hoặc 1)

**Dropout (Inference):**

Tại inference, tất cả neurons đều active, nhưng output được scale:

```
y_test = p × y_train
```

Hoặc áp dụng inverted dropout (phổ biến hơn):

**Training:**
```
ỹ_i = (r_i × y_i) / p
```

**Inference:**
```
y_test = y_train  # Không cần scale
```

**Ví dụ:**

Giả sử có 4 neurons với outputs [0.5, 0.8, 0.3, 0.9], p=0.5

**Training iteration 1:**
```
Random mask: [1, 0, 1, 0]
Output: [0.5/0.5, 0, 0.3/0.5, 0] = [1.0, 0, 0.6, 0]
```

**Training iteration 2:**
```
Random mask: [0, 1, 1, 1]
Output: [0, 0.8/0.5, 0.3/0.5, 0.9/0.5] = [0, 1.6, 0.6, 1.8]
```

**Inference:**
```
Output: [0.5, 0.8, 0.3, 0.9]  # Tất cả neurons active
```

**Tác dụng:**
- **Ensemble effect:** Mỗi iteration training với một sub-network khác nhau
- **Co-adaptation prevention:** Neurons không phụ thuộc lẫn nhau
- **Regularization:** Giảm overfitting

**Dropout2d (Spatial Dropout):**

Thay vì dropout từng neuron, dropout toàn bộ feature maps:

```
For each channel k:
    r_k ~ Bernoulli(p)
    ỹ[:,:,k] = r_k × y[:,:,k]
```

Phù hợp cho CNN vì features trong cùng channel có correlation không gian cao.

### 2.2.7. Loss Functions

**Cross-Entropy Loss (Multi-class Classification):**

```
L = -Σᵢ yᵢ × log(ŷᵢ)
```

Trong đó:
- yᵢ: True label (one-hot encoded)
- ŷᵢ: Predicted probability (from softmax)

**Ví dụ 4-class classification:**

True label: Class 1 → y = [0, 1, 0, 0]
Predicted: ŷ = [0.1, 0.7, 0.15, 0.05]

```
L = -(0×log(0.1) + 1×log(0.7) + 0×log(0.15) + 0×log(0.05))
  = -log(0.7)
  = 0.357
```

**PyTorch Implementation:**

```python
loss = nn.CrossEntropyLoss()
```

Note: PyTorch's CrossEntropyLoss đã bao gồm softmax, input là logits (trước softmax).

**Weighted Cross-Entropy (Class Imbalance):**

```
L = -Σᵢ wᵢ × yᵢ × log(ŷᵢ)
```

Trong đó wᵢ là weight cho class i:

```
wᵢ = n_samples / (n_classes × n_samples_class_i)
```

**Ví dụ:**
- Class 0: 656 samples → w₀ = 2630/(4×656) = 1.003
- Class 1: 650 samples → w₁ = 2630/(4×650) = 1.010
- Class 2: 664 samples → w₂ = 2630/(4×664) = 0.992
- Class 3: 660 samples → w₃ = 2630/(4×660) = 0.995

**Gradient của Cross-Entropy Loss:**

Kết hợp với softmax:

```
∂L/∂zᵢ = ŷᵢ - yᵢ
```

Trong đó zᵢ là logits (input của softmax).

Công thức này rất đơn giản và hiệu quả cho backpropagation!

### 2.2.8. Optimization Algorithms

**Stochastic Gradient Descent (SGD):**

```
θ_{t+1} = θ_t - η × ∇L(θ_t)
```

Trong đó:
- θ: Parameters (weights, biases)
- η: Learning rate
- ∇L(θ): Gradient của loss

**Momentum:**

```
v_t = β × v_{t-1} + (1-β) × ∇L(θ_t)
θ_{t+1} = θ_t - η × v_t
```

- β: Momentum coefficient (thường 0.9)
- Giúp tăng tốc trong hướng consistent, giảm oscillation

**Adam (Adaptive Moment Estimation):**

```
m_t = β₁ × m_{t-1} + (1-β₁) × ∇L(θ_t)        # First moment (mean)
v_t = β₂ × v_{t-1} + (1-β₂) × (∇L(θ_t))²    # Second moment (variance)

m̂_t = m_t / (1 - β₁^t)                       # Bias correction
v̂_t = v_t / (1 - β₂^t)

θ_{t+1} = θ_t - η × m̂_t / (√v̂_t + ε)
```

Hyperparameters:
- β₁ = 0.9: Exponential decay for first moment
- β₂ = 0.999: Exponential decay for second moment
- ε = 1e-8: Numerical stability
- η = 0.001: Learning rate

**AdamW (Adam with Weight Decay):**

```
θ_{t+1} = θ_t - η × (m̂_t / (√v̂_t + ε) + λ × θ_t)
```

Trong đó λ là weight decay coefficient (L2 regularization).

**Ưu điểm của Adam:**
- Adaptive learning rates cho mỗi parameter
- Hoạt động tốt với sparse gradients
- Ít sensitive với learning rate choice
- Fast convergence

**Learning Rate Scheduling:**

**ReduceLROnPlateau:**

```
if val_loss không giảm sau patience epochs:
    η_new = factor × η_old
```

Ví dụ: factor=0.5, patience=5
- Sau 5 epochs val_loss không cải thiện → η giảm 50%

### 2.2.9. Backpropagation Algorithm

**Forward Pass:**

```
Layer 1: z₁ = W₁×x + b₁,  a₁ = f₁(z₁)
Layer 2: z₂ = W₂×a₁ + b₂, a₂ = f₂(z₂)
...
Output: y = softmax(zₙ)
Loss: L = -Σ yᵢ×log(ŷᵢ)
```

**Backward Pass (Chain Rule):**

```
∂L/∂Wₙ = (∂L/∂zₙ) × (∂zₙ/∂Wₙ)
       = (∂L/∂zₙ) × aₙ₋₁^T

∂L/∂bₙ = ∂L/∂zₙ

∂L/∂aₙ₋₁ = Wₙ^T × (∂L/∂zₙ)

∂L/∂zₙ₋₁ = (∂L/∂aₙ₋₁) ⊙ f'(zₙ₋₁)
```

Trong đó ⊙ là element-wise multiplication.

**Backprop cho Convolutional Layer:**

Forward:
```
O = I * K + b
```

Backward:
```
∂L/∂K = I * (∂L/∂O)         # Gradient w.r.t. kernel
∂L/∂b = Σᵢⱼ (∂L/∂O)ᵢⱼ       # Gradient w.r.t. bias
∂L/∂I = (∂L/∂O) * K_flipped # Gradient w.r.t. input
```

**Computational Graph:**

```
Input → Conv → BN → ReLU → MaxPool → ... → FC → Softmax → Loss
  ↑                                                          ↓
  ←──────────── Backpropagation ──────────────────────────
```

**Vanishing/Exploding Gradient Problem:**

Trong deep networks:

```
∂L/∂W₁ = (∂L/∂zₙ) × (∂zₙ/∂zₙ₋₁) × ... × (∂z₂/∂z₁) × (∂z₁/∂W₁)
```

Nếu mỗi term < 1 → gradient vanish (tiến về 0)
Nếu mỗi term > 1 → gradient explode (tiến về ∞)

**Giải pháp:**
- ReLU activation (gradient = 1 for x > 0)
- Batch Normalization (ổn định gradient)
- Residual connections (skip connections)
- Gradient clipping (giới hạn gradient magnitude)

---

## 2.3. Phương pháp phân loại ảnh viễn thám

### 2.3.1. Pixel-based vs Patch-based Classification

**Pixel-based Classification:**

Mỗi pixel được phân loại độc lập dựa trên vector đặc trưng:

```
x_i = [f₁, f₂, ..., f_n]  # n features tại pixel i
y_i = classifier(x_i)      # Predict class
```

**Ưu điểm:**
- Đơn giản, dễ implement
- Nhanh (parallel processing)
- Phù hợp với ML truyền thống (RF, SVM)

**Nhược điểm:**
- Không sử dụng spatial context
- Salt-and-pepper noise trong kết quả
- Bỏ qua relationships giữa neighboring pixels

**Patch-based Classification:**

Trích xuất patches (windows) xung quanh mỗi pixel:

```
P_i = extract_patch(I, center=(row_i, col_i), size=k×k)
y_i = classifier(P_i)
```

**Ví dụ với patch 3×3:**

```
Image I:
[1  2  3  4  5]
[6  7  8  9  10]
[11 12 13 14 15]
[16 17 18 19 20]
[21 22 23 24 25]

Patch tại (2, 2):
[7  8  9 ]
[12 13 14]
[17 18 19]
```

**Ưu điểm:**
- Sử dụng spatial context
- Kết quả smooth hơn
- Phù hợp với CNN (automatic feature learning)

**Nhược điểm:**
- Chậm hơn (nhiều data hơn)
- Cần xử lý edge pixels
- Redundant computation (overlapping patches)

### 2.3.2. Spatial Autocorrelation

**Tobler's First Law of Geography:**

"Everything is related to everything else, but near things are more related than distant things."

**Moran's I (Spatial Autocorrelation Index):**

```
I = (N/W) × (Σᵢ Σⱼ wᵢⱼ(xᵢ - x̄)(xⱼ - x̄)) / (Σᵢ (xᵢ - x̄)²)
```

Trong đó:
- N: Number of samples
- wᵢⱼ: Spatial weight (1 if neighbors, 0 otherwise)
- xᵢ, xⱼ: Values at locations i, j
- x̄: Mean value

**Interpretation:**
- I > 0: Positive autocorrelation (similar values cluster)
- I ≈ 0: Random distribution
- I < 0: Negative autocorrelation (dissimilar values cluster)

**Implication for Machine Learning:**

Training và test samples gần nhau trong không gian có high correlation → **Data leakage** → Overestimate accuracy.

**Giải pháp: Stratified Data Splitting + Cross Validation**

### 2.3.3. Chiến lược chia dữ liệu

**Stratified Random Splitting:**

```
1. Stratified shuffle all samples (giữ tỷ lệ lớp)
2. Train+Val: 80% stratified samples
3. Test: 20% stratified samples (fixed)
4. 5-Fold Stratified CV trên Train+Val
```

**Ưu điểm:**

```
- Đảm bảo phân bố lớp đồng đều trong tất cả các folds
- 5-Fold CV đánh giá variance của mô hình
- Fixed test set để đánh giá khả năng tổng quát hóa
```

**Cross Validation Strategy:**

```
1. Stratified K-Fold (K=5) để chia Train+Val
2. Mỗi fold: 80% train, 20% validation
3. Báo cáo mean ± std accuracy qua 5 folds
```

**Kết quả:**

```
  Train cluster: ●●●●


  Test cluster:           ○○○○

  Distance > 50m → Independent
```

### 2.3.4. Evaluation Metrics

**Confusion Matrix:**

```
                Predicted
              0      1
Actual  0   [TN    FP]
        1   [FN    TP]
```

**Accuracy:**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision (Positive Predictive Value):**

```
Precision = TP / (TP + FP)
```

Tỷ lệ predictions đúng trong số các positive predictions.

**Recall (Sensitivity, True Positive Rate):**

```
Recall = TP / (TP + FN)
```

Tỷ lệ actual positives được detect đúng.

**F1-Score (Harmonic Mean):**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Specificity (True Negative Rate):**

```
Specificity = TN / (TN + FP)
```

**ROC Curve (Receiver Operating Characteristic):**

Plot TPR vs FPR at various thresholds:

```
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
```

**AUC (Area Under ROC Curve):**

```
AUC = ∫₀¹ TPR(FPR) d(FPR)
```

- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random classifier
- AUC > 0.9: Excellent
- 0.8 < AUC < 0.9: Good
- 0.7 < AUC < 0.8: Fair

**Multi-class Extension:**

**One-vs-Rest (OvR) AUC:**

```
AUC_ovr = (1/C) × Σᵢ AUC_i
```

Trong đó AUC_i là AUC cho class i vs all other classes.

**Macro-averaged Metrics:**

```
Precision_macro = (1/C) × Σᵢ Precisionᵢ
Recall_macro = (1/C) × Σᵢ Recallᵢ
F1_macro = (1/C) × Σᵢ F1ᵢ
```

**Weighted-averaged Metrics:**

```
Precision_weighted = Σᵢ (nᵢ/N) × Precisionᵢ
```

Trong đó nᵢ là số samples của class i.

---

**[Kết thúc Chương 2]**

📚 **Xem danh sách đầy đủ tài liệu tham khảo:** [REFERENCES.md](REFERENCES.md)
