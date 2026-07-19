# Báo cáo — Cốt lõi của Deep Learning qua bộ "Lesson materials" (RHUL)

*Kiểm tra toàn diện 15 notebook trong `Docs/Lesson materials/`. Mục tiêu: rút ra core của DL, quy trình cơ bản nhất, bản chất được dạy, và bộ visualize "chìa khoá".*

---

## 0. Kiểm kê — có gì trong thư mục

15 notebook, chia thành một mạch dạy có chủ đích: **xây bằng tay → chuyển sang PyTorch → áp dụng lên các kiến trúc → mở hộp đen → tái sử dụng**.

| Nhóm | Notebook | Ý tưởng MỚI mà nó thêm vào |
|---|---|---|
| **Nền tảng (làm bằng tay)** | Week1Ex1 — GradientDescent | Loss là một **mặt phẳng** J(m,c); học = **đi xuống dốc** theo gradient. Learning rate, momentum, RMSProp. |
| | Week1Ex2 — SimpleDNN | Model "sâu" đơn giản nhất `ŷ = w₁·w₂·x` → **độ sâu sinh ra bất ổn định**; regularisation chữa nó. |
| | Week1Ex3 — NN_Backpropagation | Backprop **viết tay** đầy đủ: forward pass + backward pass + **gradient checking** (số học vs giải tích). |
| **Chuyển sang PyTorch** | Week2Ex1 — Intro_to_pytorch | Tensor, `requires_grad`, đồ thị tính toán, `.backward()`, `.grad` — **autograd thay backprop tay**. |
| | Week2Ex2 — Backpropagation_with_pytorch | **"Cấu trúc chương trình DL cơ bản"**: `nn.Module` + loss + optimizer + vòng lặp train. |
| **Kiến trúc** | Week3Ex1 — MNIST_CNN | Ráp lại pipeline chuẩn từng mảnh; chuẩn hoá input, softmax→cross-entropy, `save/load`. |
| | Week3Ex2 — Percolation_CNN | **Inductive bias**: conv thắng dense ở bài toán không gian; "Golden Rules" của model search; **learning curve** (power law). |
| | visualize_activation | **Interpretability**: filter/feature map, activation maximization. |
| | transfer_learning_tutorial | **Tái sử dụng** ResNet-18: finetune vs feature-extractor (freeze). |
| **Chuỗi (RNN)** | 1 — recurrent_nets_learn | RNN/LSTM **giữ trạng thái qua thời gian** (bài copy nhớ xa). |
| | 2 — RNN_with_brackets | RNN học **luật đệ quy** (ngoặc cân bằng — văn phạm phi ngữ cảnh). |
| | 3 — IMDB_sentiment | RNN trên **text thật**; **RNN thua / LSTM thắng** (vanishing gradient). |
| **Biểu diễn & sinh** | MNIST_Siamese | **Metric learning**: chia sẻ trọng số + **contrastive loss**; giám sát trên **quan hệ cặp**. |
| | CGANs | **Sinh dữ liệu** bằng **trò chơi đối kháng** 2 mạng; điều kiện hoá bằng label embedding. |
| **Tổng hợp** | CourseWorkProject | Ráp mọi thứ: overfit → regularise → learning curve → kiến trúc tốt hơn. |

Ngoài ra: `Answers/` (Q1–Q6.pdf, Q7), model đã train sẵn `cifar-10-...pt`, và các hình minh hoạ CNN.

**Nhận xét tổng thể về chất lượng code:** code sư phạm rất sạch — luôn *xây từ nguyên lý trước, rồi mới gọi thư viện*, chú thích dày, luôn nhắc "nhìn dữ liệu trước", "kiểm tra shape", "kiểm tra cân bằng lớp". Có vài chỗ dùng API cũ (xem mục 7).

---

## 1. CORE của DL — chỉ có **một vòng lặp** duy nhất

Bỏ hết vỏ ngoài (CNN, RNN, GAN...), toàn bộ khoá học quy về **một ý** lặp đi lặp lại:

> **Học = tối ưu hoá.** Đoán → đo sai → tính hướng sửa → sửa một chút. Lặp lại.

Bốn bước, không hơn:

```
1. DỰ ĐOÁN     ŷ = model(x)                  (forward pass)
2. ĐO SAI       J  = loss(ŷ, y)               (một con số duy nhất)
3. TÍNH HƯỚNG   ∂J/∂w  cho mọi trọng số w     (backward pass / autograd)
4. SỬA          w ← w − η · ∂J/∂w             (một bước gradient descent)
```

Notebook đầu tiên (Week1Ex1) dạy đúng cái này ở dạng trần trụi nhất — khớp một đường thẳng `ŷ = m·x + c`:

```python
for n in range(n_iterations):
    m_grad, c_grad = J_gradient(x_data, y_data, m, c)   # bước 3
    m = m - learning_rate * m_grad                       # bước 4
    c = c - learning_rate * c_grad
```

Mọi thứ về sau chỉ là **thay ruột của 4 bước này**: model phức tạp hơn (bước 1), loss khác đi (bước 2), cách tính gradient tự động hoá (bước 3), optimizer khôn hơn (bước 4). **Vòng lặp thì không đổi.**

---

## 2. PROCESS cơ bản nhất — "cấu trúc chương trình DL"

Chính notebook Week2Ex2 đặt tên cho nó: **"Basic structure of deep learning program"**, gồm 2 phần: **SETUP** và **TRAINING**. Đây là bộ khung mà *mọi* notebook sau đều dùng lại gần như y hệt.

```python
# ── SETUP ──────────────────────────────────────────────
class Net(nn.Module):                 # 1) MODEL: kế thừa nn.Module
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 24)
        self.layer2 = nn.Linear(24, 1)
    def forward(self, x):             #    forward = định nghĩa bước DỰ ĐOÁN
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

net       = Net()
loss_fn   = nn.BCELoss()              # 2) LOSS: cách ĐO SAI
optimizer = torch.optim.RMSprop(net.parameters())   # 3) OPTIMIZER: cách SỬA

# ── TRAINING (vòng lặp) ────────────────────────────────
for epoch in range(n_epochs):
    optimizer.zero_grad()             # xoá gradient cũ (nếu không sẽ cộng dồn!)
    output = net(X_tensor)            # forward   → DỰ ĐOÁN
    loss   = loss_fn(output, Y)       #            → ĐO SAI
    loss.backward()                   # backward  → TÍNH HƯỚNG (autograd)
    optimizer.step()                  #            → SỬA trọng số
```

Và một phần thứ 3 luôn đi kèm — **đánh giá** (không tính gradient):

```python
with torch.no_grad():                 # tắt autograd cho nhanh/nhẹ
    outputs = net(X_test)
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
```

**Nhịp `zero_grad → forward → loss → backward → step`** là câu thần chú của cả khoá. Nếu chỉ được nhớ một thứ về "process của DL", nhớ nhịp 5 bước này.

Ba thứ khác nhau giữa các bài chỉ là **các nút xoay** cắm vào cùng bộ khung:

| Nút xoay | Các lựa chọn xuất hiện trong khoá |
|---|---|
| Loss | `BCELoss` (2 lớp) · `CrossEntropyLoss` (nhiều lớp) · **contrastive** (metric) · **adversarial BCE** (GAN) |
| Đầu ra | sigmoid (nhị phân) · softmax-logits (nhiều lớp) · khoảng cách Euclid (Siamese) |
| Optimizer | SGD · Momentum · **RMSProp** · Adam |
| Kiến trúc | Linear · **Conv2d/MaxPool/BatchNorm** · **LSTM/RNN** · Siamese · Generator+Discriminator |

---

## 3. BẢN CHẤT được dạy — 4 sự thật cốt lõi

Khoá học không chỉ dạy "cách gọi PyTorch". Nó dạy **bốn sự thật** làm nên bản chất DL:

### 3.1. Học là đi xuống một mặt phẳng lồi lõm (loss surface)
Week1Ex1 vẽ hẳn J(m,c) bằng `imshow` rồi thả gradient descent chạy trên đó. Bài học kèm theo: learning rate quá cao → **bay khỏi đáy**; mặt phẳng "hẹp một chiều" → dao động → cần **momentum / RMSProp**. Đây là lý do các optimizer tồn tại — không phải phép màu, mà là cách đi xuống dốc cho ổn định.

### 3.2. Backprop = quy tắc chuỗi (chain rule), được tự động hoá
Week1Ex3 bắt sinh viên **tự viết** forward + backward, rồi **kiểm tra gradient bằng số học**:

```python
estimated_grad = (J(w+δ) - J(w-δ)) / (2δ)   # so với gradient giải tích
```

Rồi Week2Ex1 cho thấy PyTorch làm đúng việc đó tự động: chỉ cần `requires_grad=True`, gọi `.backward()`, đọc `.grad`. Thông điệp: **autograd không phải hộp đen huyền bí — nó chỉ là chain rule bạn vừa tự làm bằng tay, chạy trên đồ thị tính toán.**

### 3.3. Độ sâu sinh ra bất ổn — regularisation là để chữa
Week1Ex2 là viên ngọc sư phạm: model "sâu" ngớ ngẩn nhất `ŷ = w₁·w₂·x`. Về mặt hàm số nó *y hệt* `ŷ = w·x`, nhưng vì có tích hai trọng số, **một trọng số có thể phình to trong khi trọng số kia teo lại** (miễn tích = 1) → gradient nổ → mất ổn định. Thêm phạt `λ(w₁² + w₂²)` là kéo trọng số về nhỏ → ổn định lại. Đây là **trực giác gốc của regularisation**, trước cả khi nói tới overfitting.

### 3.4. Khoảng cách giữa "học thuộc" và "hiểu" — generalization gap
Đây là **linh hồn** của phần cuối khoá (Percolation + CourseWorkProject), và cũng là nơi khoá học chạm đúng đề tài luận văn của bạn:

- **Overfit trước đã:** làm model đủ mạnh để loss trên *train* tụt xuống dưới mục tiêu, trong khi loss trên *validation* vẫn cao hơn. Khoảng chênh đó = generalization gap.
- **Rồi mới regularise:** quét nhiều mức L2/dropout, vẽ *val error theo mức regularise* → tìm điểm tối ưu (quá ít vô dụng, quá nhiều hại).
- **Learning curve:** vẽ *val error theo kích thước tập train* trên trục **log-log**; tìm xem error giảm theo **luỹ thừa** nào của N (lý tưởng ~ `1/√n`).

> **Liên hệ luận văn:** CourseWorkProject dạy *chính xác* thứ luận văn của bạn đang mổ xẻ — validation error như một hàm của N, và khoảng cách giữa hiệu năng đo được và sự thật. Khoá học coi đây là "khoa học thực nghiệm": chẩn đoán underfit/overfit, quét regularise, đo scaling — chứ không phải đoán mò kiến trúc. Đó là nền chung với chủ đề "GAP tăng theo N" của bạn.

### 3.5. (Hệ quả) Kiến trúc là biến số; vòng lặp là bất biến
Xuyên suốt 15 notebook, thứ *thay đổi* để giải bài mới **luôn là inductive bias của kiến trúc + hình học của loss**, chứ không phải vòng lặp train:

- **Conv** = chia sẻ trọng số cục bộ → hợp bài toán không gian (Percolation, ảnh).
- **LSTM** = giữ trạng thái qua thời gian → hợp chuỗi (nhớ xa, ngoặc, text).
- **Siamese** = chia sẻ trọng số 2 nhánh → học *biểu diễn/khoảng cách* thay vì nhãn.
- **GAN** = hai mạng đối kháng → *sinh* dữ liệu thay vì phân loại.

Bản chất: **"cùng một cỗ máy gradient, chỉ khác hình học của mục tiêu."**

---

## 4. VISUALIZE — chìa khoá để "nhìn thấy" DL

Khoá này coi **hình ảnh là công cụ hiểu chính**, không phải trang trí. Câu trong Week1Ex1 nói thẳng: *"Cách duy nhất để biết chuyện gì đang xảy ra là visualise các mảng số này."* Có **6 kiểu visualize cốt lõi**, xếp theo "bạn đang nhìn cái gì":

| # | Nhìn cái gì | Kỹ thuật | Dạy ở đâu |
|---|---|---|---|
| 1 | **Mặt phẳng loss** | `plt.imshow(J_grid)` — J theo 2 tham số | Week1Ex1, Ex2 |
| 2 | **Đường quyết định** | quét lưới điểm, `imshow(prediction)` + chấm dữ liệu | Week1Ex3, Week2Ex2 |
| 3 | **Trọng số & gradient** | histogram / heatmap của weight, `imshow` gradient qua các bước | Week1Ex3, Week2Ex2 |
| 4 | **Đường học (learning curve)** | `plot(loss theo epoch)`; `loglog(val_error theo N)` | Percolation, CourseWork |
| 5 | **Feature map & filter** | forward hook đọc activation; **activation maximization** | visualize_activation |
| 6 | **Trạng thái ẩn RNN** | `imshow` hidden state theo từng ký tự | RNN notebooks 1–2 |

**Điểm "aha" lớn nhất (visualize_activation):** cùng một cỗ máy gradient dùng để *train* có thể **quay ngược lại để hiểu mạng**. Thay vì lấy gradient theo *trọng số*, ta lấy gradient theo *ảnh đầu vào*:

```python
# Giữ nguyên trọng số. Bắt đầu từ ảnh nhiễu, đi LÊN dốc để làm filter kêu to nhất:
grads = torch.autograd.grad(loss_value, input_img_data)[0]   # ∂(activation)/∂(input)
grads /= (torch.sqrt(torch.mean(grads**2)) + 1e-8)           # chuẩn hoá gradient
input_img_data += grads * step                               # gradient ASCENT lên ẢNH
```

Kết quả là "ảnh mà filter đó thèm nhất" → cho thấy filter học được template gì (cạnh, đường cong, rồi tới bộ phận phức tạp). Song song đó, `register_forward_hook` cho phép đọc feature map thật của một ảnh cụ thể qua từng lớp — thấy mạng "nhìn" con ngựa/con chó như thế nào.

> Nếu phải chọn **một** hình để đại diện "chìa khoá của DL": **mặt phẳng loss với vệt gradient descent bò xuống đáy** (kiểu #1). Nó gói trọn cả core (tối ưu), cả bản chất (bề mặt lồi lõm), cả lý do cần learning rate/momentum.

---

## 5. Bản đồ toàn khoá — mạch chảy một chiều

```
   LÀM BẰNG TAY                  DÙNG PYTORCH                 ÁP DỤNG KIẾN TRÚC
 ┌───────────────┐            ┌────────────────┐          ┌──────────────────────┐
 │ Ex1 GD trên   │            │ Ex1 autograd   │          │ CNN (MNIST/Percol.)  │
 │     mặt loss  │  ───────▶  │ Ex2 khung DL   │ ──────▶  │ RNN/LSTM (nhớ/text)  │
 │ Ex2 sâu→bất ổn│            │  chuẩn         │          │ Siamese (metric)     │
 │ Ex3 backprop  │            │                │          │ GAN (sinh dữ liệu)   │
 │  + grad check │            │                │          │ Transfer (tái dùng)  │
 └───────────────┘            └────────────────┘          └──────────┬───────────┘
                                                                     │
                                                          ┌──────────▼───────────┐
                                                          │  CourseWorkProject   │
                                                          │ overfit → regularise │
                                                          │ → learning curve →   │
                                                          │   kiến trúc tốt hơn  │
                                                          └──────────────────────┘
```

Mạch dạy có một logic rõ: **hiểu cơ chế bằng tay → tin tưởng công cụ tự động → thay ruột để giải bài khó dần → cuối cùng biến thành nhà thực nghiệm** (chẩn đoán, quét, đo scaling).

---

## 6. Ba câu trả lời gọn cho câu hỏi của bạn

- **Core của DL là gì?** → Một vòng lặp tối ưu 4 bước: *đoán → đo sai (loss) → tính gradient → sửa trọng số*. Tất cả phần còn lại là biến thể của bốn bước này.
- **Process cơ bản nhất?** → Khung `nn.Module` + loss + optimizer, chạy nhịp **`zero_grad → forward → loss → backward → step`**, rồi eval trong `torch.no_grad()`.
- **Bản chất được dạy?** → (1) học = đi xuống loss surface; (2) backprop = chain rule tự động; (3) độ sâu → bất ổn → cần regularise; (4) khoảng cách "thuộc bài vs hiểu bài" (generalization gap, learning curve, power law). Kiến trúc là biến số; vòng lặp là bất biến.
- **Visualize chìa khoá?** → 6 kiểu (mục 4), mà biểu tượng nhất là **mặt phẳng loss + vệt gradient descent**, và tinh tế nhất là **lấy gradient theo đầu vào** để nhìn thấy filter học được gì.

---

## 7. Kiểm tra code — vấn đề kỹ thuật cần lưu ý

Code chạy được về mặt logic, nhưng có vài chỗ dùng **API cũ/deprecated** (do notebook viết đã lâu). Nếu chạy trên PyTorch mới cần sửa:

| File | Vấn đề | Sửa |
|---|---|---|
| visualize_activation, MNIST_CNN | `dataiter.next()` | dùng `next(dataiter)` (bản mới bỏ `.next()`) |
| visualize_activation | `from torch.autograd import Variable` + `Variable(...)` | bỏ `Variable`; đặt `requires_grad=True` thẳng trên tensor |
| visualize_activation | `torch.load('...pt')` load **cả model** (pickle) | nên lưu/nạp `state_dict()` cho bền qua các phiên bản |
| visualize_activation | `%pylab inline` | dùng `%matplotlib inline` + import tường minh |
| IMDB_sentiment | `torchtext` API (`build_vocab_from_iterator`, `get_tokenizer`) | torchtext đã đổi/deprecate ở bản mới — kiểm tra phiên bản |
| MNIST_Siamese | markdown ghi "in Keras" nhưng code là PyTorch | chỉ là chú thích cũ, vô hại |

**Chỗ nên soi kỹ (khả năng có bug nhẹ):** trong `visualize_activation`, hàm `visualize_filters_for_class` lấy `num_filters = output_images.shape[2]` *sau khi* `np.squeeze(activation[layer], axis=0)`. Activation do hook lưu là đầu ra thô của lớp conv, shape `(1, C, H, W)`, squeeze còn `(C, H, W)` → `shape[0]=C` (số kênh/filter), `shape[2]=W` (chiều rộng không gian). Vòng lặp lại đếm filter bằng `shape[2]` = **W**, không phải **C**. Trùng hợp là `conv1` cho ảnh 32×32 và có đúng 32 kênh nên `W==C=32`, chạy lọt; nhưng các lớp sâu hơn (`conv2` 64 kênh · `conv3`/`conv4` 128 kênh) có `W≠C` → **đếm sai số filter, chỉ vẽ được một phần**. **Nên chạy thử và xác nhận** trước khi tin vào hình của các lớp sâu.

*(Các con số cụ thể như shape đã suy luận từ định nghĩa `CNNModel`; nên chạy lại một ô để kiểm chứng — đúng tinh thần "gradient checking" mà chính khoá học dạy.)*

---

---

## 8. Phụ lục — Sổ tay chi tiết (key để tái sử dụng)

Phần này gom các "thẻ tra cứu" cho từng khối kiến thức của khoá học (245 thẻ). Mỗi thẻ có: **Là gì · Ý nghĩa & mục đích · Code · Visual · Lưu ý · Nguồn** — để sau này tra nhanh và tái sử dụng (cho luận văn hoặc code mới). Nội dung đã được đối chiếu lại với notebook gốc và gắn cảnh báo cho các API cũ.

**Mục lục phụ lục**

- **8.1** Tối ưu hoá & Gradient Descent
- **8.2** Backpropagation & Autograd
- **8.3** Định nghĩa Model & Hàm kích hoạt
- **8.4** Hàm mất mát (Loss functions)
- **8.5** Pipeline dữ liệu (Dataset/DataLoader/transforms)
- **8.6** CNN — khối xây dựng
- **8.7** RNN / Mô hình chuỗi
- **8.8** Metric Learning & GAN
- **8.9** Transfer Learning & Vệ sinh huấn luyện
- **8.10** Regularisation & Generalisation (lõi luận văn)
- **8.11** Kỹ thuật Visualize

---

### 8.1 Tối ưu hoá & Gradient Descent

#### Hàm mất mát / Sai số bình phương trung bình (Loss function / Mean Squared Error - MSE)

- **Là gì:** Hàm đo độ lệch giữa dự đoán yhat và giá trị thật y. MSE = trung bình của bình phương sai số: J = (1/n) Σ (yhat_i - y_i)^2.
- **Ý nghĩa & mục đích:** Là thước đo 'khớp tốt tới đâu' để tối ưu hoá bám vào. Muốn dự đoán tốt thì tìm tham số làm J nhỏ nhất. Bình phương sai số phạt lỗi lớn nặng hơn và cho hàm trơn, dễ lấy đạo hàm.
- **Code:**

```python
def calculate_J(x, y, m, c):
    yhat = m * x + c
    errs = (yhat - y) ** 2
    J = np.mean(errs)
    return J
```
- **Visual:** Vẽ colormap của J trên lưới (m, c) bằng plt.imshow: chỗ tối là J thấp (khớp tốt). Trong khoá dùng imshow(J_grid.transpose(), origin='lower', extent=[...]) rồi plt.colorbar().
- **Lưu ý:** MSE thô rất nhạy với thang đo của x (xem thẻ 'độ nhạy theo tỉ lệ'); nhân x lên 10 làm bề mặt lỗi méo hẳn. Notebook dùng NumPy thuần, không dùng nn.MSELoss của PyTorch.
- *Nguồn: Week1Exercise1_GradientDescent.txt*

#### Loss có hệ số 1/2 (Half squared loss)

- **Là gì:** Biến thể của loss cho 1 điểm dữ liệu: J = (1/2)(yhat - y)^2. Thêm 1/2 để khi lấy đạo hàm số 2 bị triệt tiêu.
- **Ý nghĩa & mục đích:** Dùng khi làm SGD trên từng điểm để đạo hàm gọn (không lòi ra hệ số 2). Bản chất giống MSE, chỉ khác hằng số nên nghiệm tối ưu không đổi.
- **Code:**

```python
# J = 0.5 * (yhat - y)**2
yhat = w * x
grad = (yhat - y) * x   # dJ/dw, đã gọn nhờ hệ số 1/2
```
- **Visual:** Với 1 tham số w, vẽ J theo w là một parabol; đáy parabol là w tối ưu. Có thể plot bằng cách quét w và tính J.
- **Lưu ý:** Đừng lẫn: hệ số 1/2 làm gradient nhỏ đi 2 lần so với MSE thô, nên learning_rate phù hợp cũng khác.
- *Nguồn: Week1Exercise2_SimpleDNN.txt*

#### Gradient của loss (Analytic gradient)

- **Là gì:** Đạo hàm của J theo từng tham số, tính bằng tay. Với đường thẳng yhat=mx+c và J là MSE: dJ/dm = mean(2(yhat-y)x), dJ/dc = mean(2(yhat-y)).
- **Ý nghĩa & mục đích:** Gradient cho biết hướng dốc lên của loss; đi ngược lại là hướng giảm loss nhanh nhất. Là thành phần cốt lõi để cập nhật tham số trong gradient descent.
- **Code:**

```python
def J_gradient(x, y, m, c):
    yhat = m * x + c
    c_grads = 2 * (yhat - y)
    m_grads = 2 * (yhat - y) * x
    c_grad = np.mean(c_grads)
    m_grad = np.mean(m_grads)
    return (m_grad, c_grad)
```
- **Visual:** Tại một điểm (m,c) trên colormap của J, gradient là mũi tên chỉ hướng dốc lên; bước cập nhật đi ngược mũi tên. Kiểm tra được: J_gradient tại nghiệm tối ưu (khoảng m=0.6, c=1.1) gần bằng 0.
- **Lưu ý:** Phải khớp công thức gradient với đúng định nghĩa J (có/không có 1/2, mean vs sum). Sai hằng số là chạy vẫn được nhưng learning_rate lệch. Ở đây tự tính gradient bằng tay, không dùng autograd của PyTorch.
- *Nguồn: Week1Exercise1_GradientDescent.txt*

#### Gradient Descent (batch, full-data)

- **Là gì:** Thuật toán lặp: mỗi bước tính gradient trên TOÀN BỘ dữ liệu rồi trừ tham số đi learning_rate nhân gradient. Lặp lại tới khi hội tụ.
- **Ý nghĩa & mục đích:** Cách chuẩn để tìm tham số làm loss nhỏ nhất khi không có nghiệm đóng, hoặc mô hình phức tạp. Dùng gradient trung bình cả tập nên hướng đi mượt, ít nhiễu.
- **Code:**

```python
m, c = 0.0, 0.0
learning_rate = 0.034
n_iterations = 200
for n in range(1, n_iterations):
    m_grad, c_grad = J_gradient(x_data, y_data, m, c)
    m = m - learning_rate * m_grad
    c = c - learning_rate * c_grad
```
- **Visual:** Lưu m_path, c_path rồi vẽ đường đi trên colormap 1/J: plt.plot(m_path, c_path, 'r') và 'r.'. Thấy quỹ đạo trượt dần về đáy loss, thường zigzag theo khe hẹp.
- **Lưu ý:** learning_rate quá cao -> quỹ đạo nảy/phân kỳ ra vô cực; quá thấp -> hội tụ rất chậm. Batch GD chậm khi dữ liệu lớn (mỗi bước quét cả tập).
- *Nguồn: Week1Exercise1_GradientDescent.txt*

#### Stochastic Gradient Descent (SGD, cập nhật từng điểm)

- **Là gì:** Biến thể của GD: cập nhật tham số ngay sau mỗi điểm dữ liệu (dùng gradient của riêng điểm đó) thay vì trung bình cả tập.
- **Ý nghĩa & mục đích:** Nhanh và mở rộng tốt cho dữ liệu lớn; cập nhật liên tục. Nhiễu ngẫu nhiên của từng điểm giúp thoát điểm kẹt nhưng làm tham số dao động quanh nghiệm tối ưu.
- **Code:**

```python
w = 1.0
learning_rate = 0.01
ww = [w]
for x, y in zip(xx, yy):
    yhat = w * x
    grad = (yhat - y) * x
    w = w - learning_rate * grad
    ww.append(w)
```
- **Visual:** plt.plot(ww): đường w theo số bước, dao động quanh giá trị tối ưu (1.0). learning_rate đủ nhỏ thì dao động hẹp và ổn định; tăng lên thì biên độ dao động lớn dần rồi mất ổn định.
- **Lưu ý:** Vì mỗi bước chỉ nhìn 1 điểm nên w không đứng yên tại nghiệm mà 'rung' quanh đó. Đặt learning_rate quá cao sẽ khiến rung biến thành phân kỳ. Đây là SGD 'thuần' (batch size 1), khác DataLoader/optim.SGD của PyTorch.
- *Nguồn: Week1Exercise2_SimpleDNN.txt*

#### Learning rate (tốc độ học)

- **Là gì:** Hằng số nhân với gradient để quyết định độ dài mỗi bước cập nhật: param = param - learning_rate * grad.
- **Ý nghĩa & mục đích:** Điều khiển sự đánh đổi giữa tốc độ và ổn định. Là siêu tham số quan trọng nhất cần chỉnh khi tối ưu.
- **Code:**

```python
learning_rate = 0.034
m = m - learning_rate * m_grad
```
- **Visual:** Chạy nhiều learning_rate và vẽ chồng các quỹ đạo trên colormap loss, hoặc vẽ J theo số vòng lặp cho từng lr: lr vừa -> J giảm nhanh và mượt; lr quá cao -> J nảy lên; lr quá thấp -> J giảm ì ạch.
- **Lưu ý:** Có một ngưỡng lr tối đa để còn hội tụ; vượt ngưỡng là phân kỳ. Ngưỡng này phụ thuộc thang đo dữ liệu và độ cong của bề mặt loss, không cố định.
- *Nguồn: Week1Exercise1_GradientDescent.txt*

#### Phân kỳ do learning rate cao (Divergence / instability)

- **Là gì:** Hiện tượng khi bước học quá lớn, mỗi cập nhật vọt qua đáy loss và ngày càng xa, tham số phóng ra vô cực thay vì hội tụ.
- **Ý nghĩa & mục đích:** Cần nhận diện để chọn learning_rate an toàn. Đây là 'pathology' kinh điển của gradient descent mà bài tập yêu cầu quan sát trực tiếp (task 1-2).
- **Code:**

```python
# Tăng learning_rate rồi in ra path để thấy phân kỳ
n_iterations = 200
learning_rate = 0.034  # thử tăng dần tới khi m_path, c_path bay đi
# tìm learning_rate lớn nhất mà quá trình còn hội tụ
```
- **Visual:** Vẽ quỹ đạo trên colormap: khi ổn định -> xoắn về đáy; khi phân kỳ -> các điểm nảy ngày một xa. Hoặc plot J theo iteration thấy J tăng vọt.
- **Lưu ý:** Với mô hình w1*w2, instability đến chậm: sai số lớn dần từ từ rồi 'catastrophic growth' đột ngột ra vô cực - dễ tưởng ổn rồi bất ngờ nổ.
- *Nguồn: Week1Exercise1_GradientDescent.txt*

#### Gradient Descent với Momentum (đà)

- **Là gì:** Biến thể giữ lại một phần thay đổi ở bước trước để tạo 'đà': m_change = momentum*m_change - lr*m_grad, rồi m = m + m_change.
- **Ý nghĩa & mục đích:** Giúp vượt qua các khe hẹp/zigzag và tăng tốc theo hướng nhất quán, giảm dao động vuông góc. Sửa được bệnh đi chậm của GD thường trên bề mặt loss lệch tỉ lệ.
- **Code:**

```python
m_change = 0.0
c_change = 0.0
for n in range(1, n_iterations):
    m_grad, c_grad = J_gradient(x_data, y_data, m, c)
    m_change = momentum * m_change - learning_rate * m_grad
    c_change = momentum * c_change - learning_rate * c_grad
    m = m + m_change
    c = c + c_change
```
- **Visual:** So sánh quỹ đạo trên colormap: GD thường zigzag trong khe, GD+momentum lướt thẳng hơn về đáy. Vẽ chồng hai path để thấy khác biệt.
- **Lưu ý:** momentum quá lớn (gần 1) có thể làm vọt qua đáy và dao động; đây là bài challenge (task 4), API tự viết bằng NumPy chứ không dùng optim.SGD(momentum=...).
- *Nguồn: Week1Exercise1_GradientDescent.txt*

#### RMSProp (chuẩn hoá bước theo độ lớn gradient)

- **Là gì:** Thuật toán tối ưu chia bước học cho căn của trung bình động bình phương gradient, để mỗi tham số có bước hiệu dụng riêng.
- **Ý nghĩa & mục đích:** Sửa bệnh GD bị một hướng (vd m) nhạy hơn hướng khác (vd c) nên phải chọn lr thoả hiệp. RMSProp tự cân bằng, ổn định hơn khi bề mặt loss lệch tỉ lệ mạnh.
- **Code:**

```python
# Tự implement (challenge 8). Ý tưởng:
# eps = 1e-8; beta = 0.9; ms = 0.0
ms = beta * ms + (1 - beta) * grad ** 2
param = param - learning_rate * grad / (np.sqrt(ms) + eps)
```
- **Visual:** Vẽ quỹ đạo RMSProp trên cùng colormap loss và so với GD: kỳ vọng RMSProp không bị bất ổn như GD thô. Đối chiếu J theo iteration.
- **Lưu ý:** Đây là challenge mở, không có code sẵn trong khoá - phải tự viết. Cần eps nhỏ ở mẫu để tránh chia cho 0.
- *Nguồn: Week1Exercise1_GradientDescent.txt*

#### Đường đồng mức / bề mặt lỗi (Loss surface / error landscape)

- **Là gì:** Hình dung loss J như một mặt phẳng độ cao trên không gian tham số; quét lưới tham số, tính J từng ô, rồi vẽ colormap hoặc contour.
- **Ý nghĩa & mục đích:** Giúp 'thấy' vì sao tối ưu dễ/khó: khe hẹp, hướng nhạy, nhiều nghiệm tương đương. Là công cụ chẩn đoán trực quan cho hành vi gradient descent.
- **Code:**

```python
J_grid = np.zeros([m_values.size, c_values.size])
for mi in range(m_values.size):
    for ci in range(c_values.size):
        J_grid[mi, ci] = calculate_J(x_data, y_data, m_values[mi], c_values[ci])
plt.imshow(J_grid.transpose(), origin='lower', extent=[0, 1, 0, 2])
plt.colorbar()
```
- **Visual:** imshow của J (hoặc 1/J để đáy sáng lên dễ nhìn). Challenge gợi ý dùng contour plot của (w1*w2-1)^2 để thấy khe nghiệm. Nhớ transpose và origin='lower' để trục đúng.
- **Lưu ý:** J thô nhiều khi khó đọc (dải giá trị rộng) -> khoá dùng 1/J_grid cho dễ thấy đáy. Đừng quên .transpose() và extent nếu không trục m/c bị hoán/đảo.
- *Nguồn: Week1Exercise1_GradientDescent.txt*

#### Mô hình 'deep' tối giản w1*w2 (over-parameterisation)

- **Là gì:** Mô hình yhat = w1*w2*x: hai trọng số nhân nhau nhưng thực chất tương đương w=w1*w2. Là 'mạng sâu' đơn giản nhất để lộ bất ổn.
- **Ý nghĩa & mục đích:** Cho thấy dư tham số tạo ra vô số nghiệm tương đương (chỉ cần w1*w2=1) và trong đó có những nghiệm trọng số cực lớn/cực nhỏ gây bất ổn khi học. Là động cơ để cần regularisation.
- **Code:**

```python
def calculate_J(x, y, w1, w2):
    yhat = w1 * w2 * x
    errs = (yhat - y) ** 2
    return np.mean(errs)

# cập nhật dùng chain rule:
# w1 = w1 - lr * grad * w2
# w2 = w2 - lr * grad * w1
```
- **Visual:** colormap/contour của J trên (w1,w2): đáy là một đường cong hyperbol w1=1/w2 (khe nghiệm), không phải một điểm. Vẽ ww1 theo ww2 để thấy trượt dọc khe rồi nổ.
- **Lưu ý:** Bất ổn đến muộn và đột ngột: nếu một trọng số phình to thì gradient của trọng số kia phình theo -> catastrophic growth. Dùng ww1[:limit] để cắt trước khi bay ra vô cực.
- *Nguồn: Week1Exercise2_SimpleDNN.txt*

#### Chain rule cập nhật nhiều trọng số (gradient qua nhiều lớp)

- **Là gì:** Với yhat=w1*w2*x, đạo hàm theo mỗi trọng số dùng quy tắc dây chuyền: dyhat/dw1=w2*x, dyhat/dw2=w1*x; nên grad chung nhân thêm trọng số kia.
- **Ý nghĩa & mục đích:** Đây là hạt nhân của lan truyền ngược (backprop): gradient của một lớp phụ thuộc giá trị các lớp khác. Giải thích vì sao trọng số lớn làm gradient lớn -> bất ổn.
- **Code:**

```python
for x, y in zip(xx, yy):
    yhat = w1 * w2 * x
    grad = (yhat - y) * x
    w1 = w1 - learning_rate * grad * w2
    w2 = w2 - learning_rate * grad * w1
```
- **Visual:** Vẽ w1*w2 theo bước (np.array(ww1)*np.array(ww2)) - đúng ra phải luôn ~1; thấy nó lệch dần rồi vọt. Cho thấy tích trọng số mới là đại lượng có nghĩa.
- **Lưu ý:** Trong vòng lặp, w1 được cập nhật TRƯỚC rồi mới dùng (w1 mới) để update w2 - đúng theo code khoá; đây là chi tiết dễ bỏ sót khi tái hiện.
- *Nguồn: Week1Exercise2_SimpleDNN.txt*

#### Regularisation L2 / weight decay (phạt trọng số lớn)

- **Là gì:** Thêm số hạng phạt λ(w1^2+w2^2) vào loss để ép trọng số nhỏ. Loss mới: J = (1/2)((yhat-y)^2 + λ(w1^2+w2^2)).
- **Ý nghĩa & mục đích:** Khi có nhiều nghiệm 'tốt như nhau', regularisation chọn nghiệm trọng số nhỏ -> ổn định hơn, tránh cặp trọng số lớn/nhỏ gây nổ. Đổi độ chính xác lấy độ ổn định.
- **Code:**

```python
regulariser = 0.01
for x, y in zip(xx, yy):
    yhat = w1 * w2 * x
    grad = (yhat - y) * x
    w1 = w1 - learning_rate * grad * w2 - regulariser * w1
    w2 = w2 - learning_rate * grad * w1 - regulariser * w2
```
- **Visual:** Vẽ ww1, ww2 theo bước: có regularisation thì hai trọng số bị kéo về gần nhau và không bay đi. Cũng plot yy vs yhats để thấy độ khớp giảm nếu λ quá lớn.
- **Lưu ý:** λ quá lớn -> kéo trọng số về 0 quá mạnh, dự đoán tệ (yhat lệch hẳn khỏi y); quá nhỏ -> không đủ chặn bất ổn. Phải dò để cân bằng. Số hạng -regulariser*w chính là 'weight decay'. Lưu ý code khoá trừ thẳng regulariser*w (không nhân learning_rate), và update w2 dùng grad*w1 (w1 vừa cập nhật).
- *Nguồn: Week1Exercise2_SimpleDNN.txt*

#### Lưu quỹ đạo tham số để chẩn đoán (parameter path logging)

- **Là gì:** Idiom: trong vòng lặp học, append giá trị tham số vào list mỗi bước để sau đó vẽ ra và quan sát hành vi hội tụ.
- **Ý nghĩa & mục đích:** Không nhìn được quá trình học nếu không ghi lại. Ghi path rồi vẽ là cách chuẩn để phát hiện dao động, zigzag, hay phân kỳ.
- **Code:**

```python
m_path = [m]
c_path = [c]
for n in range(1, n_iterations):
    # ... cập nhật m, c ...
    m_path.append(m)
    c_path.append(c)
plt.plot(m_path, c_path, 'r.')
```
- **Visual:** plot(path) theo iteration (1 tham số) hoặc plot(path_a, path_b) (2 tham số, chồng lên colormap loss). Cũng nên ghi luôn J mỗi bước để vẽ J giảm theo số vòng lặp.
- **Lưu ý:** Với mô hình dễ nổ, cắt path bằng [:limit] trước khi vẽ, không thì giá trị vô cực làm hỏng trục đồ thị.
- *Nguồn: Week1Exercise1_GradientDescent.txt*

#### Độ nhạy theo tỉ lệ dữ liệu (feature scaling / conditioning)

- **Là gì:** Loss nhạy với tham số nhân vào biến có thang lớn hơn nhiều so với tham số khác; nhân x lên 10 làm bề mặt lỗi méo và khó tối ưu.
- **Ý nghĩa & mục đích:** Giải thích vì sao J nhạy với m (hệ số của x) hơn c (hằng số): x có phương sai lớn. Đây là lý do nên chuẩn hoá dữ liệu và động lực dùng momentum/RMSProp.
- **Code:**

```python
# Thử nhân thang dữ liệu và quan sát bề mặt lỗi
x_data_scaled = 10 * x_data
# rồi dựng lại J_grid và chạy gradient descent như cũ
```
- **Visual:** Dựng lại colormap J với x đã nhân 10: khe loss hẹp và dài hơn theo hướng m -> gradient descent zigzag mạnh, cần lr nhỏ hơn để không nổ.
- **Lưu ý:** Cùng learning_rate nhưng đổi thang x có thể chuyển từ hội tụ sang phân kỳ. Bề mặt lỗi lệch tỉ lệ (ill-conditioned) là gốc rễ nhiều khó khăn của GD thô.
- *Nguồn: Week1Exercise1_GradientDescent.txt*

#### Sinh dữ liệu tổng hợp có nhiễu (synthetic data + noise)

- **Là gì:** Tạo dữ liệu giả từ một quan hệ đã biết cộng nhiễu ngẫu nhiên, ví dụ y = 1.1 + 0.6*x + noise hoặc y = x + noise, để kiểm chứng thuật toán tối ưu.
- **Ý nghĩa & mục đích:** Khi biết trước tham số 'thật' (m=0.6, c=1.1 hoặc w=1.0), ta có mốc để đánh giá tối ưu có hội tụ đúng chỗ không. Là cách chuẩn để test một pipeline học trước khi dùng dữ liệu thật.
- **Code:**

```python
x_data = np.array([x for x in range(0, 10)])
y_data = 1.1 + 0.6 * x_data + np.random.randn(x_data.size)

# hoặc dữ liệu tập trung quanh 0:
xx = 10 * (np.random.random([1000]) - 0.5)
yy = xx + 0.5 * np.random.randn(1000)
```
- **Visual:** plt.plot(x_data, y_data, '.'): đám mây điểm bám quanh đường thẳng thật; biên độ tản = mức nhiễu. Luôn check .shape rồi plot trước khi học.
- **Lưu ý:** np.random.randn (chuẩn) khác np.random.random (đều 0..1) - dễ nhầm cú pháp: randn(n) nhận số chiều rời, random([n]) nhận list kích thước. Không set seed thì mỗi lần chạy dữ liệu khác nhau.
- *Nguồn: Week1Exercise2_SimpleDNN.txt*


---

### 8.2 Backpropagation & Autograd

#### Lượt truyền xuôi (Forward pass)

- **Là gì:** Bước tính output của mạng từ input: đi từ input qua từng lớp (nhân trọng số + cộng bias + qua hàm kích hoạt) cho tới lớp output.
- **Ý nghĩa & mục đích:** Đây là bước 'dự đoán'. Phải chạy xong forward pass thì mới có output để tính loss, và mới lưu được các giá trị trung gian (inputs/outputs của từng lớp) để lượt truyền ngược dùng lại.
- **Code:**

```python
def forward_pass(self, X):
    self.X = X.copy()
    self.inputs_layer_1 = np.dot(self.X, self.w_1) + self.bias_1
    self.outputs_layer_1 = relu_outputs(self.inputs_layer_1)
    self.inputs_layer_2 = np.dot(self.outputs_layer_1, self.w_2) + self.bias_2
    self.outputs_layer_2 = sigmoid_outputs(self.inputs_layer_2)
    self.predicted_y = self.outputs_layer_2[0,0]
```
- **Visual:** Vẽ sơ đồ chuỗi: X -> [w1,b1] -> relu -> [w2,b2] -> sigmoid -> predicted_y. Hoặc dùng plot_nn_predictions: chạy forward pass trên lưới điểm và imshow bề mặt dự đoán để thấy mạng chia mặt phẳng thế nào.
- **Lưu ý:** Phải LƯU lại các giá trị trung gian (outputs_layer_1, outputs_layer_2...) trong forward pass, vì backward pass cần chúng. X có thể là view/slice nên phải .copy() để tránh bị sửa ngoài ý muốn.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Lượt truyền ngược (Backward pass / Backpropagation)

- **Là gì:** Bước tính gradient của loss theo từng trọng số, bằng cách áp dụng quy tắc chuỗi (chain rule) ngược từ output về input.
- **Ý nghĩa & mục đích:** Cho biết mỗi trọng số nên tăng hay giảm để loss nhỏ đi. Là trái tim của việc học: không có gradient thì không cập nhật được trọng số. Tận dụng lại giá trị trung gian đã lưu ở forward pass nên tính rất hiệu quả.
- **Code:**

```python
def backward_pass(self):
    self.dJ_d_outputs_layer_2[0,0] = self.dJ_d_predicted_y
    self.dJ_d_inputs_layer_2 = sigmoid_gradients(self.outputs_layer_2) * self.dJ_d_outputs_layer_2
    self.dJ_d_w_2 = np.dot(self.outputs_layer_1.transpose(), self.dJ_d_inputs_layer_2)
    self.dJ_d_outputs_layer_1 = np.dot(self.dJ_d_inputs_layer_2, self.w_2.transpose())
    self.dJ_d_inputs_layer_1 = relu_gradients(self.outputs_layer_1) * self.dJ_d_outputs_layer_1
    self.dJ_d_w_1 = np.dot(self.X.transpose(), self.dJ_d_inputs_layer_1)
```
- **Visual:** Vẽ cùng sơ đồ forward nhưng mũi tên chỉ ngược lại, ghi gradient chảy trên mỗi cạnh: dJ/dy -> dJ/d_inputs2 -> dJ/dw2 & dJ/d_outputs1 -> ... Mỗi bước = nhân gradient tới với đạo hàm cục bộ.
- **Lưu ý:** Thứ tự các phép tính phải đúng theo chiều ngược. Dễ sai chỗ transpose (dùng .transpose() sai chiều là lỗi shape). Luôn kiểm tra lại bằng gradient checking (thẻ riêng). Đây là bản viết TAY bằng numpy; ở PyTorch việc này do autograd (.backward()) tự làm.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Quy tắc chuỗi cho gradient trọng số (Chain rule: dJ/dw)

- **Là gì:** Gradient của loss theo trọng số một lớp = tích ma trận giữa (output của lớp trước, chuyển vị) và (gradient theo input của lớp đó).
- **Ý nghĩa & mục đích:** Đây là công thức lõi tái dùng cho MỌI lớp: dJ_dw = input_lớp.T @ dJ_d_inputs. Hiểu nó thì tự viết được backprop cho mạng bao nhiêu lớp cũng được.
- **Code:**

```python
# gradient theo trong so = (input cua lop).T  dot  (gradient theo tong-input cua lop)
self.dJ_d_w_2 = np.dot(self.outputs_layer_1.transpose(), self.dJ_d_inputs_layer_2)
self.dJ_d_w_1 = np.dot(self.X.transpose(), self.dJ_d_inputs_layer_1)
```
- **Visual:** Hình phép nhân ma trận: cột trái là input lớp, hàng phải là gradient input, kết quả là ma trận cùng shape với ma trận trọng số. Đối chiếu shape để nhớ: dJ_d_w_1 phải cùng shape với w_1.
- **Lưu ý:** Gradient phải CÙNG shape với ma trận trọng số tương ứng, nếu khác shape là công thức/transpose sai. Nhầm thứ tự nhân (input.T @ grad thay vì grad @ input.T) là lỗi thường gặp.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Gradient lan về lớp trước (dJ/d_outputs của lớp dưới)

- **Là gì:** Để backprop tiếp về lớp trước, nhân gradient theo input của lớp hiện tại với trọng số (chuyển vị) để ra gradient theo output của lớp trước.
- **Ý nghĩa & mục đích:** Đây là bước 'đẩy gradient ngược qua trọng số', khác với bước tính gradient CỦA trọng số. Nó cho phép tín hiệu lỗi truyền về sâu trong mạng.
- **Code:**

```python
# day gradient qua trong so, ve output cua lop truoc
self.dJ_d_outputs_layer_1 = np.dot(self.dJ_d_inputs_layer_2, self.w_2.transpose())
```
- **Visual:** Mũi tên ngược đi xuyên qua khối trọng số w_2: gradient vào từ phía output, ra phía input, nhân với w_2.T.
- **Lưu ý:** Đừng lẫn với dJ_d_w (gradient của trọng số). Ở đây w là hằng số, ta lan gradient QUA nó về activation lớp dưới. Sai transpose w là hỏng cả chuỗi phía sau.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Cặp hàm kích hoạt xuôi/ngược (Activation forward & backward)

- **Là gì:** Mỗi hàm kích hoạt cần HAI hàm: hàm forward tính output từ input, và hàm gradient tính đạo hàm cục bộ để nhân vào gradient khi truyền ngược.
- **Ý nghĩa & mục đích:** Backprop cần đạo hàm cục bộ của mỗi phi tuyến. Tách thành cặp forward/backward giúp code module hoá: đổi hàm kích hoạt chỉ cần thay cặp hàm, phần còn lại giữ nguyên.
- **Code:**

```python
def relu_outputs(inputs):        # forward: leaky ReLU
    return np.maximum(inputs,0) + 0.05 * np.minimum(inputs,0)
def relu_gradients(outputs):     # backward: dao ham cuc bo
    return (outputs > 0).astype(float) + 0.05 * (outputs < 0).astype(float)

def sigmoid_outputs(inputs):
    return 1.0 / (1.0 + np.exp(-inputs))
def sigmoid_gradients(outputs):  # sigmoid'(x) = s(1-s)
    return outputs * (1.0 - outputs)
```
- **Visual:** Vẽ 2 đường: hàm kích hoạt (S-cong cho sigmoid, gãy khúc cho leaky ReLU) và đường đạo hàm của nó bên dưới. Thấy rõ chỗ đạo hàm sigmoid gần 0 ở hai đầu (vanishing).
- **Lưu ý:** sigmoid_gradients nhận OUTPUT (đã qua sigmoid) chứ không phải input, vì dùng công thức s(1-s). Sigmoid ở 2 đầu có gradient ~0 -> dễ vanishing gradient. Leaky ReLU dùng hệ số 0.05 để tránh 'neuron chết' hoàn toàn.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Hàm mất mát và gradient của nó (Squared loss + dJ/d_predicted_y)

- **Là gì:** Loss bình phương J = 0.5*(y_pred - y)^2; đạo hàm của nó theo y_pred đơn giản là (y_pred - y).
- **Ý nghĩa & mục đích:** Loss là điểm khởi đầu của backward pass: nó vừa tính giá trị lỗi J (để theo dõi), vừa tính gradient đầu tiên dJ/d_predicted_y để đẩy ngược. Loss được viết module (tính cả J lẫn gradient) để dễ thay bằng loss khác.
- **Code:**

```python
def squared_loss(self, y):
    self.J = 0.5 * (self.predicted_y - y)**2
    self.dJ_d_predicted_y = self.predicted_y - y
```
- **Visual:** Parabol J theo (y_pred - y): đáy ở 0 khi dự đoán đúng. Đạo hàm là đường thẳng đi qua gốc, dấu cho biết đẩy y_pred lên hay xuống.
- **Lưu ý:** Hệ số 0.5 là để đạo hàm ra gọn (y_pred - y), không có ý nghĩa khác. Phải gọi squared_loss NGAY SAU forward_pass và TRƯỚC backward_pass vì backward dùng dJ_d_predicted_y.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Gradient của bias (dJ/d_bias)

- **Là gì:** Gradient của loss theo bias bằng đúng gradient theo tổng-input của neuron đó, vì bias chỉ cộng thẳng vào input.
- **Ý nghĩa & mục đích:** Nhắc rằng bias không cần công thức riêng phức tạp: đạo hàm của (input + bias) theo bias là 1, nên dJ/d_bias = dJ/d_inputs. Tiết kiệm suy nghĩ khi tự viết backprop.
- **Code:**

```python
# bias duoc cong vao input cua neuron, nen dJ_d_bias = dJ_d_inputs:
self.total_dJ_d_bias_2 += self.dJ_d_inputs_layer_2
self.total_dJ_d_bias_1 += self.dJ_d_inputs_layer_1
```
- **Visual:** Sơ đồ neuron: mũi tên bias cộng thẳng vào nút tổng, ghi 'đạo hàm = 1' trên cạnh đó, nên gradient đi qua nguyên vẹn.
- **Lưu ý:** Với minibatch, gradient bias lớp 1 là cả vector [1, n_layer_1] (mỗi neuron một bias). Đừng nhầm nó với gradient theo trọng số (là ma trận). Trong source, dJ_d_inputs_layer_2 CHÍNH là dJ_d_bias_2 (chỉ một biến, không tính lại).
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Tích luỹ gradient theo batch (Gradient accumulation / batching)

- **Là gì:** Cộng dồn gradient của nhiều ví dụ vào biến total_ trước khi cập nhật trọng số một lần, thay vì cập nhật sau mỗi ví dụ.
- **Ý nghĩa & mục đích:** Cho phép chạy chế độ batch/minibatch: chạy forward+backward cho từng ví dụ, cộng dồn gradient, rồi lấy trung bình để cập nhật. Gradient trung bình ít nhiễu hơn nên bước cập nhật ổn định hơn.
- **Code:**

```python
self.n_backward_passes += 1
self.total_J += self.J
self.total_dJ_d_w_2 += self.dJ_d_w_2
self.total_dJ_d_w_1 += self.dJ_d_w_1
self.total_dJ_d_bias_2 += self.dJ_d_inputs_layer_2
self.total_dJ_d_bias_1 += self.dJ_d_inputs_layer_1
```
- **Visual:** Thanh cộng dồn: mỗi ví dụ thêm một mũi tên gradient vào cùng một 'thùng' total, đến cuối batch mới đổ ra cập nhật.
- **Lưu ý:** Phải nhớ RESET các total về 0 sau mỗi lần cập nhật, nếu không gradient cũ dồn sang batch sau. Chia cho n_backward_passes để lấy trung bình chứ không cộng thô. Chú ý: PyTorch autograd cũng CỘNG DỒN .grad y hệt vậy, nên trong vòng lặp huấn luyện phải zero grad mỗi bước.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Cập nhật trọng số bằng gradient descent (update_weights)

- **Là gì:** Trừ trọng số đi một lượng bằng learning_rate nhân gradient trung bình: w = w - lr * grad.
- **Ý nghĩa & mục đích:** Đây là bước 'học' thực sự: dời trọng số theo hướng dốc xuống của loss. Chia learning_rate cho số ví dụ trong batch để có gradient trung bình, rồi mới trừ.
- **Code:**

```python
def update_weights(self, learning_rate):
    assert self.n_backward_passes > 0
    batch_learning_rate = learning_rate / self.n_backward_passes
    self.w_1 -= batch_learning_rate * self.total_dJ_d_w_1
    self.bias_1 -= batch_learning_rate * self.total_dJ_d_bias_1
    self.w_2 -= batch_learning_rate * self.total_dJ_d_w_2
    self.bias_2 -= batch_learning_rate * self.total_dJ_d_bias_2
    # reset
    self.n_backward_passes = 0
    self.total_J = 0
    self.total_dJ_d_w_1 *= 0; self.total_dJ_d_w_2 *= 0
    self.total_dJ_d_bias_1 *= 0; self.total_dJ_d_bias_2 *= 0
```
- **Visual:** Bóng lăn xuống thung lũng loss: mỗi update là một bước dời theo hướng âm gradient. Learning rate = độ dài bước.
- **Lưu ý:** assert n_backward_passes > 0 để tránh chia 0 (trọng số hoá Inf/NaN). Quên reset total sau update sẽ làm gradient tích luỹ sai. Dấu là TRỪ gradient (đi xuống), cộng nhầm là loss tăng. Ở PyTorch, bước này thay bằng optimizer.step() (bên trong torch.no_grad()).
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Batch vs Stochastic vs Minibatch gradient descent

- **Là gì:** Ba chế độ khác nhau ở chỗ cập nhật trọng số sau bao nhiêu ví dụ: batch = cả tập rồi mới update, stochastic = update sau mỗi ví dụ, minibatch = update sau một nhóm nhỏ.
- **Ý nghĩa & mục đích:** Chọn chế độ đánh đổi giữa độ mượt và tốc độ: batch cho gradient chính xác nhưng chậm cập nhật; stochastic nhiễu nhưng cập nhật liên tục, thoát cực tiểu cục bộ tốt hơn; minibatch dung hoà. Cùng một code chỉ đổi chỗ đặt update_weights.
- **Code:**

```python
# BATCH: cong don het roi update 1 lan
for i in range(0, X.shape[0]):
    nn.forward_pass(X[i:i+1,:]); nn.squared_loss(Y[i]); nn.backward_pass()
nn.update_weights(0.1)

# STOCHASTIC: update ngay sau moi vi du
# for i in ...: forward; loss; backward; nn.update_weights(0.1)
```
- **Visual:** Ba đường loss chồng nhau: batch mượt-chậm, stochastic răng cưa nhiều, minibatch ở giữa. Vẽ global_mean_J qua các epoch cho mỗi chế độ.
- **Lưu ý:** Vì update_weights tự chia cho n_backward_passes, chế độ nào cũng dùng learning_rate 'thô' như nhau. Stochastic có thể cần learning rate nhỏ hơn vì mỗi bước dựa trên 1 ví dụ nhiều nhiễu.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Learning rate và đường học (Learning rate / learning curve)

- **Là gì:** Learning rate là hệ số quyết định độ lớn mỗi bước cập nhật trọng số; learning curve là đường loss trung bình theo thời gian huấn luyện.
- **Ý nghĩa & mục đích:** Learning rate là siêu tham số quan trọng nhất để chỉnh: quá nhỏ thì học chậm, quá lớn thì loss dao động hoặc phân kỳ. Nhìn learning curve để biết đang học tốt, chững lại, hay nổ.
- **Code:**

```python
global_mean_J = []
for j in range(0, n_epochs):
    for i in range(0, X.shape[0]):
        nn.forward_pass(X[i:i+1,:]); nn.squared_loss(Y[i]); nn.backward_pass()
    global_mean_J.append(nn.total_J / nn.n_backward_passes)
    nn.update_weights(0.1)
plt.plot(global_mean_J)
```
- **Visual:** plt.plot(global_mean_J) cho đường loss giảm dần. plt.loglog(global_mean_J) để thấy hành vi luỹ thừa. plt.loglog(-np.diff(global_mean_J)) xem tốc độ giảm loss mỗi bước.
- **Lưu ý:** Learning rate quá cao -> loss dao động/tăng vọt/NaN (phân kỳ). Nếu đường loss phẳng ngay từ đầu, có thể lr quá nhỏ hoặc gradient sai. Dùng np.diff để soi 100 thay đổi cuối xem đã hội tụ chưa. Lưu ý: phải append(total_J/n_backward_passes) TRƯỚC update_weights vì update reset total_J về 0.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Khởi tạo trọng số ngẫu nhiên (Weight initialisation)

- **Là gì:** Đặt trọng số ban đầu là số ngẫu nhiên chuẩn (mean 0) với độ lệch chuẩn sigma, thay vì để toàn số 0.
- **Ý nghĩa & mục đích:** Nếu tất cả trọng số bằng 0 (hoặc bằng nhau), mọi neuron học giống hệt nhau (symmetry) và mạng không học được gì. Khởi tạo ngẫu nhiên phá đối xứng; sigma điều chỉnh độ lớn tín hiệu ban đầu.
- **Code:**

```python
def initialise_weights(self, sigma_1, sigma_2):
    self.w_1 = sigma_1 * np.random.randn(self.w_1.shape[0], self.w_1.shape[1])
    self.bias_1 = sigma_1 * np.random.randn(1, self.w_1.shape[1])
    self.w_2 = sigma_2 * np.random.randn(self.w_2.shape[0], self.w_2.shape[1])

nn = NN2layer(2, 16)
nn.initialise_weights(0.2, 0.2)
```
- **Visual:** Histogram trọng số ngay sau khởi tạo: hình chuông quanh 0 với bề rộng = sigma. So với khởi tạo 0 (một cột duy nhất tại 0).
- **Lưu ý:** Khởi tạo toàn 0 làm mọi neuron đối xứng, mạng chết. Sigma quá lớn -> input vào sigmoid bão hoà, gradient ~0. Sigma quá nhỏ -> tín hiệu yếu, học chậm. np.random.randn cho N(0,1) rồi nhân sigma. (Chú ý: source chỉ init w_2, bias_2 vẫn để 0 — vẫn ổn vì w_1 đã phá đối xứng.)
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Kiểm tra gradient bằng sai phân số (Numerical gradient checking)

- **Là gì:** Ước lượng gradient bằng cách nhích một trọng số lên/xuống một lượng delta nhỏ, đo loss hai bên, rồi lấy (Jplus - Jminus)/(2*delta), so với gradient mà backprop tính.
- **Ý nghĩa & mục đích:** Cách đáng tin nhất để bắt lỗi backprop. Nếu gradient giải tích (backward pass) và gradient số khớp nhau thì code backprop đúng. Tác giả nói tự tìm ra vài lỗi nhờ cách này.
- **Code:**

```python
delta = 0.01
estimated = np.zeros(nn.w_1.shape)
for r in range(nn.w_1.shape[0]):
    for c in range(nn.w_1.shape[1]):
        w = nn.w_1[r,c]
        nn.w_1[r,c] = w + delta
        nn.forward_pass(test_x); nn.squared_loss(test_y); Jplus = nn.J
        nn.w_1[r,c] = w - delta
        nn.forward_pass(test_x); nn.squared_loss(test_y); Jminus = nn.J
        nn.w_1[r,c] = w   # tra lai gia tri cu!
        estimated[r,c] = (Jplus - Jminus) / (2 * delta)
# so sanh: nn.dJ_d_w_1  vs  estimated
```
- **Visual:** Vẽ tán xạ gradient giải tích (trục x) vs gradient số (trục y): đúng thì các điểm nằm trên đường chéo y=x. Hoặc in cạnh nhau và trừ để xem sai lệch nhỏ.
- **Lưu ý:** PHẢI trả trọng số về giá trị cũ sau mỗi lần thử, nếu không làm hỏng mạng cho ô kế tiếp. Dùng sai phân TRUNG TÂM (2 bên, chia 2*delta) chính xác hơn một bên. delta quá nhỏ gây sai số làm tròn, quá lớn thì xấp xỉ kém. Ở PyTorch có torch.autograd.gradcheck làm việc tương đương.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Trực quan hoá trọng số bằng heatmap (Weight heatmap)

- **Là gì:** Xếp bias và ma trận trọng số của một lớp thành ảnh và hiển thị bằng imshow để nhìn giá trị bằng màu.
- **Ý nghĩa & mục đích:** Nhìn nhanh xem neuron nào đang làm việc, trọng số phân bố ra sao, có neuron nào 'chết' (trọng số/hoạt động không đổi) không. Trực quan hoá thuật toán số theo nhiều cách giúp phát hiện bất thường.
- **Code:**

```python
plt.imshow(np.vstack([nn.bias_1, nn.w_1]))
plt.colorbar()

# ghi lai gradient moi update, xep thanh heatmap de xem doi dau:
# row = nn.dJ_d_w_1.copy().reshape(1, -1)
# heat = np.vstack(list_of_100_row_vectors)
# plt.imshow(heat)
```
- **Visual:** imshow ma trận trọng số: mỗi ô một màu theo giá trị, colorbar bên cạnh. Xếp gradient của 100 update thành các hàng để xem gradient nào đổi dấu (màu nhảy qua lại quanh 0).
- **Lưu ý:** Phải .copy() gradient trước khi lưu lại từng bước, nếu không mọi hàng trỏ về cùng một mảng và bị ghi đè. reshape(1,-1) để biến ma trận gradient thành vector hàng trước khi np.vstack.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Kiểm tra neuron có được dùng không (Neuron activation std)

- **Là gì:** Tính output của mỗi neuron lớp ẩn trên toàn tập huấn luyện, rồi xem độ lệch chuẩn của output từng neuron.
- **Ý nghĩa & mục đích:** Nếu một neuron cho ra gần như cùng một giá trị với mọi input (std ~ 0), nó không đóng góp gì — mạng đang lãng phí công suất. Cách chẩn đoán mạng có neuron thừa/chết.
- **Code:**

```python
# gom output lop 1 cho moi vi du, roi:
# activations shape = [n_examples, n_layer_1]
# std_per_neuron = activations.std(axis=0)
# plt.hist(std_per_neuron)  hoac  plt.imshow(std_per_neuron.reshape(1,-1))
```
- **Visual:** Histogram hoặc heatmap độ lệch chuẩn output của từng neuron: neuron gần 0 là không được dùng (chết), neuron std lớn là đang phân biệt dữ liệu tốt.
- **Lưu ý:** Đây là bài tập challenge (không có code đầy đủ trong notebook) — cần tự gom activation. Neuron chết thường do khởi tạo xấu hoặc ReLU bị đẩy vào vùng âm mãi.
- *Nguồn: Week1Exercise3_NN_Backpropagation.txt*

#### Tensor PyTorch vs mảng numpy (torch.from_numpy, chia sẻ bộ nhớ)

- **Là gì:** Tensor giống mảng numpy nhưng thêm khả năng tính gradient tự động. torch.from_numpy tạo tensor DÙNG CHUNG bộ nhớ với mảng numpy gốc.
- **Ý nghĩa & mục đích:** Tensor là kiểu dữ liệu nền của PyTorch; hiểu quan hệ với numpy giúp chuyển dữ liệu qua lại. Điểm mấu chốt để dùng autograd là dữ liệu phải là tensor chứ không phải mảng numpy.
- **Code:**

```python
import torch, numpy as np
np_data = np.array([[1,2],[3,4]])
torch_data = torch.from_numpy(np_data)
torch_data[0,1] = 17
np_data   # gia tri trong numpy CUNG doi -> chung bo nho
```
- **Visual:** Hai hộp 'np_data' và 'torch_data' cùng trỏ vào một khối bộ nhớ; sửa một bên hộp kia đổi theo.
- **Lưu ý:** from_numpy chia sẻ bộ nhớ: sửa tensor thì mảng numpy cũng đổi và ngược lại — dễ gây bug bất ngờ. Muốn bản sao độc lập thì torch.tensor(np_data) (copy) hoặc .clone(). Tensor tạo từ mảng int sẽ có dtype int64, không dùng autograd được; ép .float() nếu cần gradient.
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Bật tính gradient (requires_grad)

- **Là gì:** Thuộc tính của tensor; đặt True để PyTorch lưu lại đồ thị tính toán cho các phép liên quan tới tensor đó, nhằm tính được gradient sau này.
- **Ý nghĩa & mục đích:** Đây là công tắc bật autograd. Chỉ những tensor có requires_grad=True (thường là trọng số, input cần gradient) mới được PyTorch theo dõi và tính .grad.
- **Code:**

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.rand_like(x, requires_grad=True)  # cung requires_grad
```
- **Visual:** Đánh dấu các nút trong đồ thị tính toán: nút có requires_grad=True được tô đậm (được theo dõi), nút thường thì mờ.
- **Lưu ý:** Nếu quên đặt requires_grad=True cho trọng số thì .grad sẽ là None, không học được. Tensor số nguyên không nhận requires_grad — phải là kiểu float (dùng 1.0 chứ không 1). Deprecation: KHÔNG dùng torch.autograd.Variable nữa (đã gộp vào Tensor từ PyTorch 0.4); chỉ cần đặt requires_grad trên tensor thường.
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Tính gradient tự động (.backward())

- **Là gì:** Gọi y.backward() để PyTorch đi ngược đồ thị tính toán và tính gradient của y theo mọi tensor có requires_grad=True.
- **Ý nghĩa & mục đích:** Thay cho việc tự viết backward pass bằng tay: chỉ cần forward pass bằng hàm torch rồi gọi .backward(), autograd lo phần đạo hàm. Đây là điểm mạnh chính của PyTorch.
- **Code:**

```python
a = (w * x).sum()                     # forward (co the viet sum(w*x))
y = torch.max(a, torch.tensor(0.0))   # relu
y.backward()                          # tinh moi gradient nguoc
w.grad   # dJ/dw
x.grad   # dJ/dx
```
- **Visual:** Đồ thị tính toán x,w -> a -> y; gọi backward là làn sóng gradient chảy ngược từ y về x và w, điền giá trị vào .grad.
- **Lưu ý:** y phải là tensor MỘT SỐ (scalar) thì backward() mới chạy trực tiếp; nếu y nhiều phần tử phải truyền vector gradient (y.backward(gradient=...)). Sau backward, các kết quả trung gian của forward bị xoá (trừ khi retain_graph=True). Nên dùng (w*x).sum() hoặc torch.sum(w*x) thay cho Python sum() cho rõ ràng.
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Đọc gradient đã tính (.grad)

- **Là gì:** Thuộc tính chứa gradient của tensor sau khi gọi .backward() trên một kết quả scalar phụ thuộc vào nó.
- **Ý nghĩa & mục đích:** Nơi lấy kết quả của autograd để cập nhật trọng số. Kiểm tra được công thức bằng tay: dy/dw = x * (dy/da), dy/dx = w * (dy/da).
- **Code:**

```python
y.backward()
print(w.grad)   # = x * dy/da
print(x.grad)   # = w * dy/da
```
- **Visual:** Đặt cạnh giá trị .grad với công thức giải tích tương ứng để đối chiếu (giống gradient checking nhưng của PyTorch).
- **Lưu ý:** .grad là None nếu chưa gọi backward, hoặc tensor không có requires_grad, hoặc là kết quả trung gian (đã bị xoá). Gọi backward nhiều lần sẽ CỘNG DỒN vào .grad chứ không ghi đè — nhớ reset (.grad=None hoặc optimizer.zero_grad()).
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Đồ thị tính toán (Computational graph)

- **Là gì:** Bản ghi các phép toán nối input tới output; PyTorch tự dựng nó khi ta tính bằng hàm torch trên tensor có requires_grad, để biết cách đi ngược tính gradient.
- **Ý nghĩa & mục đích:** Là cấu trúc autograd dùng để backprop tự động. Mỗi phép torch trong forward vừa tính giá trị vừa nhớ cách tính đạo hàm của nó, ghép lại thành đồ thị.
- **Code:**

```python
# moi ham torch forward deu keo theo mot ham backward
a = (w * x).sum()
y = torch.max(a, torch.tensor(0.0))
# do thi: x,w -> (nhan, cong) -> a -> (max) -> y
y.backward()  # di nguoc do thi
```
- **Visual:** Sơ đồ nút-cạnh: các tensor là nút, các phép toán là cạnh; mũi tên xuôi cho forward, mũi tên ngược cho luồng gradient.
- **Lưu ý:** Phải dùng HÀM TORCH (torch.max, ...) chứ không phải hàm numpy trên tensor, vì chỉ phép torch mới có backward gắn kèm để dựng đồ thị. Đồ thị dựng ĐỘNG (define-by-run): mỗi forward tạo đồ thị mới, và mặc định bị giải phóng sau một lần backward.
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Tensor lá vs trung gian (Leaf tensors / retain_grad)

- **Là gì:** Tensor lá là tensor ở đầu (input, trọng số) và cuối đồ thị; PyTorch mặc định chỉ giữ .grad cho tensor lá và vứt gradient của kết quả trung gian.
- **Ý nghĩa & mục đích:** Tiết kiệm bộ nhớ: thường ta chỉ cần gradient của trọng số (tensor lá) để học, không cần gradient của giá trị trung gian. Nếu vẫn muốn xem gradient trung gian thì gọi .retain_grad() trước backward.
- **Code:**

```python
a = (w * x).sum()
a.retain_grad()          # yeu cau giu gradient cua a
y = torch.max(a, torch.tensor(0.0))
y.backward()
a.grad                   # gio moi co gia tri
```
- **Visual:** Đồ thị với nút a ở giữa tô mờ (gradient bị xoá); sau retain_grad() thì nút a được tô đậm giữ lại .grad.
- **Lưu ý:** Không gọi retain_grad() thì a.grad là None (PyTorch đã vứt) và có warning. Phải gọi retain_grad() TRƯỚC backward. Sau một backward thường phải tính lại forward vì đồ thị đã bị giải phóng.
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Giữ lại đồ thị để backward nhiều lần (retain_graph)

- **Là gì:** Mặc định PyTorch xoá đồ thị (và kết quả trung gian của forward) sau một lần .backward(); truyền retain_graph=True để giữ lại và backward thêm lần nữa.
- **Ý nghĩa & mục đích:** Cần khi muốn gọi backward nhiều lần trên cùng một forward pass. Nếu không, lần backward thứ hai báo lỗi vì đồ thị đã mất.
- **Code:**

```python
# giu do thi de goi backward lai
y.backward(retain_graph=True)
# ... co the backward them lan nua
# neu khong: phai tinh lai forward truoc moi backward
```
- **Visual:** Hai lần sóng gradient chảy ngược trên cùng một đồ thị (đồ thị không bị xoá giữa hai lần).
- **Lưu ý:** Quên retain_graph rồi backward lần hai -> lỗi 'Trying to backward through the graph a second time... graph freed'. Giữ đồ thị tốn bộ nhớ nên chỉ dùng khi thực sự cần. Cách khác đơn giản hơn là tính lại forward mỗi lần.
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Gradient cộng dồn và reset trong autograd (grad accumulation / grad=None)

- **Là gì:** Mỗi lần gọi .backward(), gradient mới được CỘNG vào .grad hiện có thay vì ghi đè; muốn bắt đầu lại thì đặt .grad = None trước khi tính.
- **Ý nghĩa & mục đích:** Cộng dồn là tính năng cố ý (hữu ích khi gom gradient nhiều lần tính, như batch), nhưng nếu không muốn thì phải tự xoá. Đây là lý do trong vòng lặp huấn luyện luôn phải 'zero grad'.
- **Code:**

```python
# chay lai se CONG DON gradient vao .grad
# reset truoc khi tinh lai:
x.grad = None
w.grad = None
# ... roi forward + y.backward() moi
```
- **Visual:** Thanh .grad tăng dần mỗi lần backward nếu không reset; sau grad=None thì về 0 và bắt đầu lại.
- **Lưu ý:** Đây chính là lỗi kinh điển 'quên zero_grad': không reset thì gradient các bước dồn lại làm cập nhật sai. Trong notebook này reset bằng x.grad = None; ở code thực dùng optimizer.zero_grad() (hoặc model.zero_grad()) đầu mỗi vòng lặp.
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Vì sao phải dùng hàm torch trong forward (torch ops có backward)

- **Là gì:** Mỗi hàm forward của torch có một hàm backward đi kèm để tính đạo hàm khi lan ngược; hàm numpy/python thường không có nên phá vỡ autograd.
- **Ý nghĩa & mục đích:** Giải thích quy tắc thực hành: khi muốn autograd hoạt động, mọi phép trong forward pass phải là phép torch trên tensor. Đó là cách PyTorch biết cách đi ngược qua từng phép.
- **Code:**

```python
# DUNG: ham torch -> co backward
y = torch.max(a, torch.tensor(0.0))
# TRANH: np.maximum(a, 0) tren tensor -> mat lien ket autograd
```
- **Visual:** Mỗi phép trong forward là một khối có hai mặt: mặt xuôi (tính giá trị) và mặt ngược (tính gradient). Hàm numpy chỉ có mặt xuôi.
- **Lưu ý:** Trộn numpy vào giữa forward của tensor requires_grad làm đứt đồ thị (thường báo lỗi 'can't convert tensor that requires grad to numpy') -> gradient sai hoặc None. Luôn dùng torch.* cho phép trong luồng cần gradient.
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Các thuộc tính của tensor (shape, dtype, device, requires_grad, grad)

- **Là gì:** Bộ thuộc tính mô tả một tensor: shape (kích thước mỗi chiều), dtype (kiểu số, mặc định float32), device (cpu/gpu), requires_grad (có theo dõi gradient), grad (gradient đã tính).
- **Ý nghĩa & mục đích:** Cần nắm để debug: luôn biết shape và dtype của tensor đang dùng; biết requires_grad/grad để chắc autograd đang chạy đúng; device chỉ quan trọng khi có gpu.
- **Code:**

```python
x.shape          # = x.size(), kich thuoc
x.dtype          # float32 mac dinh (tinh nhanh)
x.device         # cpu hay gpu
x.requires_grad  # co theo doi gradient khong
x.grad           # gradient sau backward()
```
- **Visual:** Thẻ tóm tắt một tensor: một bảng nhỏ liệt kê 5 thuộc tính và giá trị hiện tại — dùng khi debug shape/kiểu.
- **Lưu ý:** float32 là mặc định và cho số học nhanh; ép sai dtype (vd int64) làm requires_grad không dùng được. y trong backward phải là scalar. x.shape và x.size() là một, dùng cái nào cũng được.
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Tạo tensor theo mẫu (rand_like / randn_like / ones_like / zeros_like)

- **Là gì:** Nhóm hàm tạo tensor MỚI cùng shape (và có thể cùng thuộc tính) với một tensor cho trước, điền giá trị theo mẫu: ngẫu nhiên đều, ngẫu nhiên chuẩn, toàn 1, toàn 0.
- **Ý nghĩa & mục đích:** Tiện khởi tạo trọng số hoặc bộ đệm cùng kích thước với dữ liệu mà không phải gõ lại shape. rand_like cho giá trị không âm (unif[0,1)), randn_like cho N(0,1).
- **Code:**

```python
w = torch.rand_like(x, requires_grad=True)   # unif[0,1), can requires_grad
# torch.randn_like(x, requires_grad=True)    # N(0,1)
# torch.ones_like(x)  / torch.zeros_like(x)
```
- **Visual:** Histogram giá trị: rand_like phẳng trong [0,1), randn_like hình chuông quanh 0, ones_like/zeros_like một cột duy nhất.
- **Lưu ý:** *_like sao chép shape (và dtype/device) nhưng requires_grad KHÔNG tự bật — phải truyền requires_grad=True nếu cần gradient. Notebook dùng rand_like thay randn_like để trọng số không âm cho ví dụ ReLU (đảm bảo a>0).
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Tắt theo dõi gradient (torch.no_grad)

- **Là gì:** Khối with torch.no_grad(): tạm ngừng dựng đồ thị tính toán — mọi phép bên trong không được autograd theo dõi.
- **Ý nghĩa & mục đích:** Dùng khi CẬP NHẬT trọng số bằng tay và khi ĐÁNH GIÁ/dự đoán: lúc đó không cần gradient nên tắt đi để tiết kiệm bộ nhớ, chạy nhanh hơn, và tránh vô tình đưa bước cập nhật vào đồ thị. Đây là cầu nối giữa update_weights viết tay (numpy) và autograd.
- **Code:**

```python
# cap nhat trong so bang tay ma khong lam ban do thi
with torch.no_grad():
    w -= learning_rate * w.grad
    w.grad = None   # zero grad cho vong sau

# hoac khi danh gia:
# with torch.no_grad():
#     preds = model(X_test)
```
- **Visual:** Một 'công tắc' bao quanh khối lệnh: trong khối, các mũi tên dựng đồ thị bị ngắt; ra khỏi khối lại bật.
- **Lưu ý:** Nếu cập nhật w -= lr*w.grad mà KHÔNG bọc trong no_grad, PyTorch sẽ báo lỗi vì đang sửa in-place một tensor lá cần gradient. Đừng nhầm no_grad (tắt theo dõi) với model.eval() (chỉ đổi hành vi Dropout/BatchNorm) — thực tế thường dùng cả hai khi test.
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Tách tensor khỏi đồ thị (.detach())

- **Là gì:** x.detach() trả về một tensor mới CÙNG dữ liệu với x nhưng ĐÃ CẮT khỏi đồ thị tính toán (requires_grad=False), nên gradient không lan qua nó.
- **Ý nghĩa & mục đích:** Dùng khi muốn lấy giá trị của một tensor để ghi log, vẽ đồ thị, hoặc chuyển sang numpy mà không muốn autograd theo dõi; hoặc khi cố ý CHẶN gradient chảy về một nhánh. Bổ trợ cho no_grad ở mức từng tensor.
- **Code:**

```python
loss_value = loss.detach()          # lay so de log, khong giu do thi
arr = x.detach().cpu().numpy()      # chuyen tensor requires_grad sang numpy
# chan gradient chay ve mot nhanh:
# y = f(a.detach())
```
- **Visual:** Một cái kéo cắt cạnh nối tensor với phần đồ thị phía trước: bản detach đứng riêng, gradient dừng ở đó.
- **Lưu ý:** Phải .detach() TRƯỚC .numpy() nếu tensor có requires_grad, nếu không PyTorch báo lỗi 'Can't call numpy() on Tensor that requires grad'. .detach() chia sẻ bộ nhớ với tensor gốc (sửa in-place bản này ảnh hưởng bản kia); muốn bản độc lập thì .detach().clone().
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*

#### Vòng lặp huấn luyện chuẩn PyTorch (zero_grad → backward → step)

- **Là gì:** Khuôn mẫu mỗi bước học trong PyTorch: xoá gradient cũ, forward tính loss, backward tính gradient, rồi optimizer.step() cập nhật trọng số.
- **Ý nghĩa & mục đích:** Đây là dạng 'công nghiệp hoá' của chu trình forward→loss→backward→update viết tay ở phần numpy. optimizer thay cho update_weights; zero_grad thay cho việc reset total gradient. Nhớ khuôn này là dùng được PyTorch cho mọi mô hình.
- **Code:**

```python
optimizer = torch.optim.SGD([w], lr=0.1)
for epoch in range(n_epochs):
    optimizer.zero_grad()      # xoa gradient cu (thay cho .grad=None)
    y = model_forward(w, x)
    loss = loss_fn(y, target)
    loss.backward()            # tinh gradient
    optimizer.step()           # cap nhat trong so
```
- **Visual:** Vòng tròn 4 mũi tên nối tiếp: zero_grad → forward/loss → backward → step, rồi quay lại zero_grad.
- **Lưu ý:** Quên zero_grad thì gradient CỘNG DỒN qua các bước -> học sai (lỗi kinh điển). optimizer.step() đã tự chạy trong ngữ cảnh không dựng đồ thị nên không cần bọc no_grad. Deprecation: dùng optimizer.zero_grad() thay cho việc gán .grad=None thủ công; và không dùng Variable() nữa.
- *Nguồn: Week2Exercise1_Intro_to_pytorch.txt*


---

### 8.3 Định nghĩa Model & Hàm kích hoạt

#### Định nghĩa mạng bằng cách kế thừa nn.Module (subclass nn.Module)

- **Là gì:** Cách chuẩn để tạo một mạng nơ-ron trong Pytorch: viết một class con của nn.Module, khai báo các lớp trong __init__ và mô tả luồng tính toán trong forward.
- **Ý nghĩa & mục đích:** Dùng khi muốn gói toàn bộ phần khởi tạo (init trọng số ngẫu nhiên) và phần forward vào MỘT chỗ. Lợi ích lớn nhất: mỗi lần cần mạng mới chỉ việc tạo instance mới, tránh lỗi kinh điển là quên reset trọng số rồi train lại nhiều lần dưới các điều kiện khác nhau -> kết quả thí nghiệm sai.
- **Code:**

```python
class Net1(nn.Module):
    def __init__(self):
        super().__init__()          # modern; tương đương super(Net1, self).__init__()
        self.layer1 = nn.Linear(2, 24)
        self.layer2 = nn.Linear(24, 1)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x
```
- **Visual:** Vẽ sơ đồ khối: input (n,2) -> [Linear 2->24] -> ReLU -> [Linear 24->1] -> Sigmoid -> output (n,1). Có thể print(net1) để Pytorch tự in cấu trúc các lớp.
- **Lưu ý:** Phải gọi super().__init__() ngay đầu __init__, nếu không nn.Module không đăng ký được tham số. Source viết super(Net1, self).__init__() (kiểu Python 2 vẫn chạy); PyTorch hiện đại chỉ cần super().__init__() gọn hơn. Quên tạo instance MỚI khi đổi điều kiện thí nghiệm -> train chồng lên trọng số cũ, kết quả rác.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt*

#### Lớp tuyến tính / fully-connected (nn.Linear)

- **Là gì:** Một lớp thực hiện phép biến đổi tuyến tính output = x @ W.T + b, với số input và số output do ta chỉ định. Tự động tạo sẵn ma trận trọng số và vector bias.
- **Ý nghĩa & mục đích:** Là viên gạch cơ bản của mạng feedforward. nn.Linear(in, out) ánh xạ vector kích thước in sang out. Ví dụ layer 2->24 nghĩa là 2 input, 24 nơ-ron; ma trận trọng số lưu dạng (24,2) cộng 24 bias.
- **Code:**

```python
self.layer1 = nn.Linear(2, 24)  # 2 inputs -> 24 neurons
self.layer2 = nn.Linear(24, 1)  # 24 inputs -> 1 neuron
# bỏ bias: nn.Linear(2, 24, bias=False)
```
- **Visual:** Với input (n,2), output ra (n,24). requires_grad mặc định True nên trọng số sẽ được cập nhật khi train. Lưu ý: .weight có shape (out_features, in_features) = (24,2), không phải (2,24).
- **Lưu ý:** Số output của lớp trước phải khớp số input của lớp sau, sai là lỗi shape. Bias mặc định có; muốn tắt phải truyền bias=False. Về mặt khái niệm là x*W+b, nhưng tensor .weight lưu theo (out,in) nên phép thực tế là x @ W.T + b.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt*

#### Hàm forward (forward pass)

- **Là gì:** Phương thức bắt buộc trong class kế thừa nn.Module, mô tả cách dữ liệu chảy qua các lớp để ra output. Gọi net1(x) sẽ tự chạy forward.
- **Ý nghĩa & mục đích:** Đây là nơi ta xâu chuỗi các lớp và chèn hàm kích hoạt giữa chúng. Backprop được Pytorch tự dựng ngầm từ chính chuỗi phép toán trong forward, nên ta chỉ cần viết chiều xuôi.
- **Code:**

```python
def forward(self, x):
    x = torch.relu(self.layer1(x))
    x = torch.sigmoid(self.layer2(x))
    return x

# gọi mạng: dùng net1(x), KHÔNG tự gọi forward thủ công
output = net1(X_tensor)
```
- **Visual:** Vẽ mũi tên nối tiếp: x -> layer1 -> relu -> layer2 -> sigmoid -> return. Mỗi bước ghi rõ shape để dễ debug.
- **Lưu ý:** Nên gọi net1(x) chứ không phải net1.forward(x) trực tiếp. Bản demo trong source có chỗ gọi nn1.forward(...) cho tiện, nhưng cách chuẩn là net1(x) để Pytorch chạy đúng các hook (__call__). Luôn theo dõi shape của tensor qua từng lớp.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt*

#### Hàm kích hoạt ReLU (ReLU activation)

- **Là gì:** Hàm kích hoạt phi tuyến: giữ nguyên giá trị dương, ép giá trị âm về 0. Có ở dạng hàm torch.relu(...) / F.relu(...) và dạng lớp nn.ReLU().
- **Ý nghĩa & mục đích:** Chèn phi tuyến vào giữa các lớp Linear để mạng học được hàm phi tuyến (như checkerboard). Không có phi tuyến thì nhiều lớp Linear xếp chồng vẫn chỉ tương đương một phép tuyến tính. ReLU là lựa chọn mặc định phổ biến vì đơn giản, học nhanh.
- **Code:**

```python
# dạng hàm (dùng trong forward)
x = torch.relu(self.layer1(x))   # hoặc F.relu(...)

# dạng lớp (dùng trong nn.Sequential)
nn.ReLU()
```
- **Visual:** Đồ thị hình gãy: y=0 khi x<0, y=x khi x>=0. Vẽ bằng plt.plot(np.linspace(-3,3,100), np.maximum(0, np.linspace(-3,3,100))).
- **Lưu ý:** Khoá học gợi ý thử so sánh tanh/sigmoid với relu để thấy khác biệt tốc độ học. torch.relu / F.relu (hàm) và nn.ReLU (lớp) là các cách viết cùng một thứ.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt*

#### Hàm kích hoạt Sigmoid (Sigmoid activation)

- **Là gì:** Hàm ép mọi số thực về khoảng (0,1) theo đường cong chữ S. Dùng qua torch.sigmoid(...).
- **Ý nghĩa & mục đích:** Đặt ở lớp output khi làm phân loại 2 lớp, để output đọc được như xác suất thuộc lớp 1. Cặp đôi tự nhiên với BCELoss (binary cross-entropy).
- **Code:**

```python
x = torch.sigmoid(self.layer2(x))  # squashes into (0,1)
```
- **Visual:** Đường cong chữ S đi từ 0 lên 1, đi qua 0.5 tại x=0. Vẽ: plt.plot(t, 1/(1+np.exp(-t))).
- **Lưu ý:** Nếu output đã qua sigmoid thì dùng nn.BCELoss. Nếu để output là logits thô (chưa sigmoid) thì dùng nn.BCEWithLogitsLoss (ổn định số học hơn, tự gộp sigmoid) — đừng gắn sigmoid rồi lại dùng BCEWithLogitsLoss. Bài phân loại nhiều lớp dùng softmax chứ không phải sigmoid.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt*

#### Hàm kích hoạt Tanh (Tanh activation)

- **Là gì:** Hàm kích hoạt phi tuyến hình chữ S nhưng ép giá trị về khoảng (-1,1). Được gợi ý như một biến thể thay cho ReLU/sigmoid.
- **Ý nghĩa & mục đích:** Khoá học đề nghị thử tanh để so sánh hiệu ứng của các hàm kích hoạt khác nhau lên tốc độ và chất lượng học. Là công cụ để thí nghiệm, không cố định một lựa chọn.
- **Code:**

```python
# thay relu bằng tanh trong forward
x = torch.tanh(self.layer1(x))
```
- **Visual:** Đường chữ S đối xứng quanh gốc, chạy từ -1 đến 1. Vẽ: plt.plot(t, np.tanh(t)).
- **Lưu ý:** Đây là gợi ý bài tập ('try using tanh or sigmoid as opposed to relu'); notebook không dùng sẵn tanh trong code chính, cần tự thay vào forward.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt*

#### Tạo instance mạng và in cấu trúc (instantiate & print model)

- **Là gì:** Sau khi định nghĩa class, tạo một đối tượng mạng bằng cách gọi tên class như hàm; mạng ra đời với trọng số ngẫu nhiên, chưa được train.
- **Ý nghĩa & mục đích:** Cần một instance cụ thể để chạy forward và train. print(net) cho Pytorch tự in cấu trúc các lớp, hữu ích để kiểm tra kiến trúc đúng như ý.
- **Code:**

```python
net1 = Net1()
print(net1)
```
- **Visual:** Output text liệt kê các lớp: Net1((layer1): Linear(in_features=2, out_features=24, bias=True) ...). Đây chính là bản 'sơ đồ' kiến trúc dạng chữ.
- **Lưu ý:** Mỗi instance mới có trọng số ngẫu nhiên riêng. Khi làm thí nghiệm so sánh, nhớ tạo instance mới để reset, đừng dùng lại mạng đã train (source nhấn mạnh đây là lỗi rất hay gặp).
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt*

#### Tham số của mạng (net.parameters())

- **Là gì:** Phương thức trả về một generator chứa toàn bộ tensor trọng số và bias của mạng (những thứ sẽ được học).
- **Ý nghĩa & mục đích:** Dùng để đưa tham số vào optimizer, hoặc để đếm/xem trọng số. requires_grad mặc định True nên mọi tham số này đều được tính gradient.
- **Code:**

```python
list(net1.parameters())  # xem toàn bộ weights & biases
# thường dùng để tạo optimizer:
optimizer = torch.optim.RMSprop(net1.parameters())
```
- **Visual:** Với Net1(2->24->1): weight (24,2), bias (24,), weight (1,24), bias (1,). Có thể in [p.shape for p in net1.parameters()] để kiểm tra.
- **Lưu ý:** parameters() trả generator (chỉ duyệt được một lần); bọc bằng list(...) nếu muốn xem lại nhiều lần. Muốn xem kèm tên lớp thì dùng named_parameters() hoặc state_dict().
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt*

#### Xây mạng nhanh bằng nn.Sequential

- **Là gì:** Constructor bậc cao xâu chuỗi một dãy lớp theo thứ tự; input chảy qua lần lượt từng lớp mà không cần tự viết forward.
- **Ý nghĩa & mục đích:** Cách nhanh nhất để dựng mạng đơn giản (tuyến tính, xếp lớp). Có thể dùng độc lập, hoặc nhúng bên trong một nn.Module để gọn code.
- **Code:**

```python
model1 = nn.Sequential(
    nn.Conv2d(1, 6, 3),
    nn.ReLU())

# hoặc lồng trong class:
self.layers = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10))
```
- **Visual:** Chuỗi hộp nối tiếp theo đúng thứ tự khai báo. Mỗi hộp là một lớp; dữ liệu đi từ trái sang phải.
- **Lưu ý:** Sequential chỉ nhận các lớp (nn.*), KHÔNG nhận hàm thuần (torch.relu) — phải dùng bản lớp nn.ReLU(). Sequential kém linh hoạt hơn subclass nn.Module: với kiến trúc có nhánh (skip-connection) thì phải tự viết forward.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Lớp làm phẳng (nn.Flatten)

- **Là gì:** Lớp duỗi một tensor nhiều chiều (ví dụ ảnh 1x28x28) thành vector một chiều (784), giữ nguyên chiều batch.
- **Ý nghĩa & mục đích:** Cần thiết để nối ảnh 2D vào một lớp nn.Linear (vốn nhận vector). Thường đặt ngay trước lớp Linear đầu tiên trong mạng phân loại ảnh.
- **Code:**

```python
self.layers = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10))  # 28*28 = 784
```
- **Visual:** Hình: ô vuông 28x28 được kéo dài thành một hàng 784 ô. Chiều batch (chiều 0) giữ nguyên.
- **Lưu ý:** 784 = 28*28 phải khớp kích thước ảnh MNIST; đổi ảnh khác kích thước phải đổi con số input của Linear theo. nn.Flatten mặc định giữ chiều 0 (batch) và duỗi từ chiều 1 trở đi.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Lớp tích chập 2D (nn.Conv2d)

- **Là gì:** Lớp tích chập cho ảnh: nn.Conv2d(in_channels, out_channels, kernel_size). Mỗi kênh ra là một bộ lọc nhỏ trượt trên ảnh.
- **Ý nghĩa & mục đích:** Dùng cho dữ liệu ảnh để học đặc trưng cục bộ (cạnh, nét) mà không phải nối đầy đủ mọi pixel. Ví dụ Conv2d(1,6,3): 1 kênh xám vào, 6 kênh ra, kernel 3x3.
- **Code:**

```python
nn.Conv2d(1, 6, 3)  # 1 input channel, 6 output channels, 3x3 kernel
# input phải có dạng (batch, channels, H, W):
output = model1(mnist_trainset[0][0].view(1, 1, 28, 28))
```
- **Visual:** Bộ lọc 3x3 trượt trên ảnh 28x28, tạo feature map 26x26 (giảm 2 mỗi chiều với kernel 3, không padding). 6 kênh ra = 6 feature map.
- **Lưu ý:** Kernel 3x3 (padding mặc định 0) làm ảnh co lại 2 pixel mỗi chiều (28->26). Input bắt buộc 4 chiều (batch, channel, H, W); ảnh đơn phải .view(1,1,28,28).
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Lớp gộp cực đại (nn.MaxPool2d)

- **Là gì:** Lớp lấy giá trị lớn nhất trong từng ô cửa sổ để thu nhỏ feature map. nn.MaxPool2d(2,2) giảm mỗi chiều đi một nửa.
- **Ý nghĩa & mục đích:** Giảm kích thước không gian (downsample) để giảm tính toán và tăng bất biến vị trí nhỏ. Thường xen kẽ sau các lớp Conv2d + ReLU.
- **Code:**

```python
nn.MaxPool2d(2, 2)  # cửa sổ 2x2, bước 2 -> giảm nửa mỗi chiều
```
- **Visual:** Chia feature map thành các ô 2x2, mỗi ô lấy 1 giá trị max; ví dụ 24x24 -> 12x12.
- **Lưu ý:** Trong notebook lớp này nằm ở phần code bị comment (gợi ý kiến trúc sâu hơn), chưa chạy trực tiếp; là công cụ để tự ghép mạng CNN sâu hơn.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Output logits thô (unnormalised outputs / logits)

- **Là gì:** Lựa chọn thiết kế: để lớp cuối trả về số thực tuỳ ý (chưa qua softmax), thay vì trả về xác suất đã chuẩn hoá.
- **Ý nghĩa & mục đích:** Linh hoạt hơn: ta có thể dùng hàm khác nhau cho lúc dự đoán và lúc tính loss. Đặc biệt, để logits thô cho phép dùng CrossEntropyLoss (gộp softmax + log-loss ổn định về số học).
- **Code:**

```python
class NN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    def forward(self, x):
        return self.layers(x)  # trả logits thô, KHÔNG softmax
```
- **Visual:** Output là vector 10 số thực có cả âm lẫn dương (ví dụ [-0.3, 1.2, ...]); chưa cộng thành 1. Chuyển thành xác suất bằng softmax để nhìn.
- **Lưu ý:** Vì output là logits, KHÔNG gắn softmax trong forward nếu bạn dùng CrossEntropyLoss (loss đó đã tự làm log-softmax; gắn thêm sẽ softmax hai lần, sai).
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Softmax để ra xác suất (F.softmax)

- **Là gì:** Hàm biến một vector số thực tuỳ ý thành phân phối xác suất (các giá trị dương, cộng lại bằng 1). Dùng F.softmax(x, dim=...).
- **Ý nghĩa & mục đích:** Là cách chuẩn để chuyển logits thô của lớp cuối thành xác suất cho phân loại nhiều lớp, để đọc/diễn giải dự đoán từng lớp.
- **Code:**

```python
import torch.nn.functional as F
tmp_probs = F.softmax(tmp, dim=1)  # dim=1: chuẩn hoá theo chiều lớp
```
- **Visual:** Vector logits -> vector xác suất; mạng mới khởi tạo ngẫu nhiên cho ra ~0.1 cho cả 10 lớp (đoán mò). Có thể bar-plot 10 xác suất.
- **Lưu ý:** Phải chọn đúng dim (dim=1 khi shape là (batch, num_classes)). Khi train thường KHÔNG softmax trong mạng mà để CrossEntropyLoss lo, chỉ dùng softmax lúc muốn xem xác suất.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Import các module Pytorch (nn, F)

- **Là gì:** Quy ước import: torch.nn (các loại lớp) và torch.nn.functional (các hàm như kích hoạt, softmax dùng trong forward).
- **Ý nghĩa & mục đích:** Phân biệt rõ: nn.* là các lớp có tham số/đối tượng (nn.Linear, nn.ReLU); F.* là các hàm thuần (F.softmax, F.relu). Biết chỗ tìm đúng công cụ khi định nghĩa mạng.
- **Code:**

```python
import torch
import torch.nn as nn            # các loại layer
import torch.nn.functional as F   # activation functions, v.v.
```
- **Visual:** Không có hình trực tiếp; hình dung hai 'hộp công cụ': nn = các lớp lắp ghép, F = các hàm biến đổi.
- **Lưu ý:** Nhiều thứ có cả bản lớp (nn.ReLU) lẫn bản hàm (F.relu/torch.relu) — chọn bản lớp cho nn.Sequential, bản hàm cho viết tay trong forward.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt*

#### Chế độ đánh giá eval() (evaluation mode)

- **Là gì:** Phương thức đặt mạng sang chế độ suy luận: các lớp như dropout và batch normalisation chuyển sang hành vi lúc dự đoán thay vì lúc train.
- **Ý nghĩa & mục đích:** Gọi trước khi dùng mạng để dự đoán/đánh giá (sau khi load lại model), đảm bảo dropout tắt và batchnorm dùng thống kê đã học, cho kết quả nhất quán.
- **Code:**

```python
nn1_reloaded.eval()  # đặt dropout & batchnorm sang chế độ inference
# quay lại huấn luyện:
nn1_reloaded.train()
```
- **Visual:** Không có hình; là một 'công tắc' chế độ train/eval của mạng.
- **Lưu ý:** eval() CHỈ đổi hành vi dropout/batchnorm, KHÔNG tắt việc tính gradient — muốn tắt gradient phải bọc thêm torch.no_grad(). Quên gọi eval() sau khi load -> dropout/batchnorm vẫn ở chế độ train -> dự đoán bị nhiễu. Nhớ .train() khi quay lại huấn luyện.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Lưu & tải lại tham số model (state_dict, torch.save / load_state_dict)

- **Là gì:** state_dict là một dict ánh xạ tên tham số -> tensor trọng số/bias của model. Lưu bằng torch.save, nạp lại bằng load_state_dict.
- **Ý nghĩa & mục đích:** Cách chuẩn để lưu một model đã train và dùng lại sau này. Khuyến nghị lưu RIÊNG state_dict (chỉ trọng số), rồi tự tạo lại instance đúng class và nạp trọng số vào — an toàn và bền hơn là lưu cả object.
- **Code:**

```python
PATH = "saved_state_dict.pt"
torch.save(nn1.state_dict(), PATH)          # lưu trọng số

nn1_reloaded = NN1()                        # phải tạo lại đúng kiến trúc
nn1_reloaded.load_state_dict(torch.load(PATH))
nn1_reloaded.eval()
```
- **Visual:** In state_dict thấy tên + shape từng tham số, ví dụ 'layers.1.weight  torch.Size([10, 784])'. Đây là 'bản kê' toàn bộ trọng số học được.
- **Lưu ý:** Deprecation/thay đổi hành vi: PyTorch >= 2.6 mặc định torch.load(weights_only=True). Nạp state_dict vẫn OK, nhưng lưu/nạp CẢ object bằng torch.save(nn1, PATH) rồi torch.load(PATH) sẽ lỗi trừ khi truyền weights_only=False — và cách lưu cả object (dựa trên pickle) dễ vỡ khi đổi code/đường dẫn class, nên ưu tiên state_dict. Bắt buộc tạo lại đúng class trước khi load_state_dict.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Tắt tính gradient khi suy luận (torch.no_grad)

- **Là gì:** Context manager tạm dừng ghi lại computational graph, để forward pass không tốn bộ nhớ/tính toán cho backprop.
- **Ý nghĩa & mục đích:** Dùng khi chỉ dự đoán/đánh giá (tính accuracy trên validation, vẽ biên quyết định) — lúc đó không cần gradient. Giúp chạy nhanh hơn và tiết kiệm bộ nhớ.
- **Code:**

```python
with torch.no_grad():
    outputs = nn1(images)
    _, predicted = torch.max(outputs, dim=1)
```
- **Visual:** Không có hình; hình dung một 'công tắc' tắt việc theo dõi gradient bên trong khối with, rồi bật lại khi ra khỏi khối.
- **Lưu ý:** no_grad() và eval() là HAI thứ độc lập: no_grad tắt autograd, eval() đổi hành vi dropout/batchnorm. Khi inference nên dùng cả hai; dùng cái này không thay được cái kia.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*


---

### 8.4 Hàm mất mát (Loss functions)

#### Hàm mất mát nhị phân BCELoss (Binary Cross-Entropy Loss)

- **Là gì:** nn.BCELoss() là hàm mất mát dùng cho phân loại 2 lớp, nhận đầu vào là MỘT xác suất đã qua sigmoid (giá trị trong khoảng 0..1) và nhãn 0/1. Nó chính là log-loss (logistic loss).
- **Ý nghĩa & mục đích:** Dùng khi mạng có đúng 1 neuron đầu ra qua sigmoid, dự đoán xác suất của lớp 1. Phù hợp bài toán 2 lớp (vd percolation: có/không có đường xuyên, ảnh trong/ngoài hình tròn). Nó phạt nặng khi mạng tự tin nhưng sai. Với 2 lớp, dùng 1 sigmoid + BCELoss gọn hơn 2 output + softmax vì tránh trọng số dư thừa.
- **Code:**

```python
import torch.nn as nn

loss_function = nn.BCELoss()

# output = sigmoid neuron, shape phai khop voi target
output = net(X_tensor)              # da qua torch.sigmoid
average_loss = loss_function(output, Y_like_output)
average_loss.backward()             # bat dau backprop tu day
```
- **Visual:** Chưa qua huấn luyện, loss xấp xỉ -log(0.5) = 0.69 (đoán ngẫu nhiên 50/50). Vẽ đường loss theo epoch để thấy nó tụt dần: plt.plot(history['train_loss'], label='train_loss'); plt.plot(history['val_loss'], label='val_loss'); plt.legend(). Nếu train_loss tụt mà val_loss đi lên là dấu hiệu overfit.
- **Lưu ý:** BCELoss yêu cầu đầu vào ĐÃ qua sigmoid (0..1) — nếu đưa giá trị thô chưa squash sẽ sai/lỗi. Shape của output và target PHẢI khớp nhau, thường phải reshape target (vd Y_tensor.reshape([100,1])) nếu không kết quả loss tính sai một cách âm thầm. Lưu ý hiện đại: BCELoss kém ổn định số học; nếu tự viết mạng nên bỏ Sigmoid ở lớp cuối và dùng nn.BCEWithLogitsLoss (xem thẻ riêng).
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt; Week3Exercise2_Percolation_CNN.txt; CGANs.txt*

#### Khớp shape target với output trước khi tính loss (reshape target)

- **Là gì:** Trước khi tính average loss, tensor nhãn Y phải có cùng shape với output của mạng. Ta reshape Y để giống output.
- **Ý nghĩa & mục đích:** Đây là lỗi rất hay gặp: output của mạng có shape (n,1) nhưng Y ban đầu là (n,). Nếu không khớp, PyTorch có thể broadcast sai và cho ra loss vô nghĩa mà không báo lỗi. Vì vậy luôn phải biết shape của tensor.
- **Code:**

```python
output = net1(X_tensor)          # shape (100,1)
Y_like_output = Y_tensor.reshape([100,1])   # cho khop
average_loss = loss(output, Y_like_output)
```
- **Visual:** In shape ra để kiểm tra: print(output.shape, Y_like_output.shape) — hai cái phải in ra giống hệt nhau.
- **Lưu ý:** Broadcasting âm thầm là cái bẫy lớn: shape (n,1) với (n,) không khớp nhưng vẫn 'chạy được' và ra số sai. Luôn check .shape hai vế trước khi tính loss.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt*

#### Log-loss / mất mát log tính thủ công (Log-loss / Negative Log-Likelihood)

- **Là gì:** Log-loss của một dự đoán là âm của log xác suất mà mạng gán cho lớp ĐÚNG: -log(p_true_class).
- **Ý nghĩa & mục đích:** Giúp hiểu bản chất của cross-entropy: nếu mạng gán xác suất cao cho lớp đúng thì loss nhỏ, gán thấp thì loss lớn. Đây là 'viên gạch' để hiểu vì sao CrossEntropyLoss/BCELoss hoạt động. Tính tay để đối chiếu với hàm loss của PyTorch.
- **Code:**

```python
tmp_probs = F.softmax(tmp, dim=1)          # doi output tho -> xac suat
tmp_probs_n = tmp_probs.detach().numpy()
# true class cua item 0 la 5
log_loss = -np.log(tmp_probs_n[0,5])
```
- **Visual:** Với 10 lớp và mạng chưa học, mỗi xác suất xấp xỉ 0.1 nên loss xấp xỉ -log(0.1) = 2.30. Có thể vẽ đường -log(p) theo p để thấy nó bùng lên vô cực khi p tiến về 0 (phạt cực nặng khi tự tin mà sai).
- **Lưu ý:** Muốn lấy giá trị số từ tensor có gradient thì phải .detach().numpy() trước, không thì lỗi. -np.log(0) là vô cực, nên trong thực tế các hàm loss chuẩn cộng thêm ổn định số học — đừng tự viết -log() thô cho production.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Mất mát Cross-Entropy đa lớp (CrossEntropyLoss)

- **Là gì:** nn.CrossEntropyLoss() là một hàm gộp softmax + log-loss trong một bước, nhận đầu vào là output THÔ (chưa chuẩn hoá, logits) và nhãn lớp dạng số nguyên (LongTensor).
- **Ý nghĩa & mục đích:** Dùng cho phân loại nhiều lớp (vd MNIST 10 chữ số). Ưu điểm: mạng xuất ra giá trị thô, ta để CrossEntropyLoss tự làm softmax rồi log-loss một cách hiệu quả và ổn định số học. Tách rời hàm dự đoán (softmax) và hàm tính loss cho linh hoạt.
- **Code:**

```python
loss_function = nn.CrossEntropyLoss()

outputs = nn1(images)                 # logits tho, KHONG qua softmax
loss = loss_function(outputs, labels) # labels la LongTensor cac chi so lop
loss.backward()
```
- **Visual:** Kiểm tra: loss thủ công -np.log(prob_true) phải bằng loss_function(tmp, torch.LongTensor([5])). Vẽ đường training loss theo epoch để theo dõi hội tụ.
- **Lưu ý:** KHÔNG tự thêm softmax trước khi đưa vào CrossEntropyLoss — nó đã bao gồm softmax rồi, thêm nữa sẽ làm hỏng gradient. Nhãn phải là số nguyên lớp (LongTensor), không phải one-hot. (PyTorch mới cũng cho phép nhãn là phân phối xác suất mềm, nhưng mặc định vẫn là chỉ số lớp nguyên.)
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Sigmoid + BCELoss so với Softmax + CrossEntropy (chọn loss cho 2 lớp)

- **Là gì:** Với bài toán 2 lớp có hai cách tương đương: (a) 1 neuron sigmoid + BCELoss, hoặc (b) 2 output + softmax + CrossEntropyLoss.
- **Ý nghĩa & mục đích:** Giúp chọn thiết kế đầu ra. Với đúng 2 lớp, cách (a) gọn hơn vì chỉ cần 1 xác suất; cách (b) có trọng số đầu ra dư thừa. Softmax với 2 lớp thực chất tương đương sigmoid. Hiểu điều này để không lãng phí tham số.
- **Code:**

```python
# Cach a: 2 lop
...
nn.Linear(64,1),
nn.Sigmoid()
loss_function = nn.BCELoss()

# Cach b: da lop (dung cho >2 lop)
nn.Linear(784,10)   # output tho, khong sigmoid
loss_function = nn.CrossEntropyLoss()
```
- **Visual:** Không có hình riêng; minh hoạ bằng bảng so sánh: cột 'số output', 'activation cuối', 'hàm loss', 'kiểu nhãn' cho hai cách.
- **Lưu ý:** Đừng lẫn lộn: BCELoss cần output đã sigmoid (0..1); CrossEntropyLoss cần output THÔ. Dùng nhầm activation với nhầm loss là lỗi rất hay xảy ra.
- *Nguồn: Week3Exercise2_Percolation_CNN.txt; Week3Exercise1_MNIST_CNN.txt*

#### Loss là trung bình trên cả batch (mean loss over batch)

- **Là gì:** Hàm loss của PyTorch tính loss từng ví dụ rồi lấy trung bình, trả về một tensor 0 chiều (một số) đại diện cho cả minibatch.
- **Ý nghĩa & mục đích:** Đây là 'lá cuối' của đồ thị tính toán mà từ đó ta backprop. Vì là trung bình nên loss không phụ thuộc mạnh vào kích thước batch, dễ so sánh. Ta cộng dồn loss từng batch rồi chia số batch để có loss trung bình mỗi epoch.
- **Code:**

```python
total_loss = 0
n_mini_batches = 0
for images, labels in trainloader:
    outputs = net(images)
    loss = loss_function(outputs, labels)   # da trung binh tren batch
    total_loss += loss.item()
    n_mini_batches += 1
epoch_loss = total_loss / n_mini_batches
```
- **Visual:** Vẽ epoch_loss theo epoch. Đầu tiên cao rồi giảm. So train_loss với val_loss trên cùng biểu đồ để chẩn đoán overfit/underfit.
- **Lưu ý:** Loss là tensor 0 chiều — KHÔNG index được, phải dùng .item() để lấy số. Nếu cộng dồn cả tensor (không .item()) sẽ giữ nguyên đồ thị gradient và ngốn bộ nhớ dần. Lưu ý deprecation: lấy minibatch đầu bằng tmpiter.next() (có trong notebook) đã lỗi thời — dùng next(iter(trainloader)) trên PyTorch hiện đại.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt; Week3Exercise2_Percolation_CNN.txt*

#### Lấy giá trị số từ loss bằng .item()

- **Là gì:** .item() chuyển một tensor 0 chiều (như loss trung bình) thành một số Python thuần.
- **Ý nghĩa & mục đích:** Loss trả về là tensor không có chiều nên không index được. Muốn in ra, cộng dồn để thống kê, hay lưu vào list để vẽ đồ thị thì phải rút giá trị số bằng .item().
- **Code:**

```python
print(i, average_loss.item())   # in loss moi buoc
total_loss += loss.item()       # cong don de tinh trung binh
```
- **Visual:** Không có hình; giá trị .item() chính là con số bạn plot theo epoch.
- **Lưu ý:** Nếu cộng dồn trực tiếp tensor loss thay vì loss.item(), đồ thị tính toán bị giữ lại qua các batch, bộ nhớ phình dần và có thể tràn.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt; Week3Exercise1_MNIST_CNN.txt*

#### Backprop từ loss bằng loss.backward()

- **Là gì:** .backward() gọi trên tensor loss sẽ lan truyền ngược, tính gradient của loss theo mọi trọng số trong mạng.
- **Ý nghĩa & mục đích:** Đây là bước biến loss thành thông tin để cập nhật trọng số. Loss là điểm xuất phát của backprop; sau backward() mọi tham số có .grad, rồi optimizer.step() dùng gradient đó để chỉnh trọng số.
- **Code:**

```python
optimizer.zero_grad()        # xoa gradient cu
outputs = net(images)
loss = loss_function(outputs, labels)
loss.backward()              # tinh moi gradient
optimizer.step()             # cap nhat trong so
```
- **Visual:** Có thể vẽ histogram gradient của một layer sau backward: plt.hist(net1.layer1.weight.grad.numpy().reshape(-1), 7) — xem gradient lớn/nhỏ ra sao.
- **Lưu ý:** PHẢI zero_grad() trước mỗi backward, nếu không gradient CỘNG DỒN qua các bước và làm hỏng cập nhật. Đây là lỗi kinh điển. Backward mặc định chỉ gọi được trên tensor loss vô hướng (0 chiều); tensor nhiều phần tử phải truyền gradient argument hoặc reduce trước.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt; Week3Exercise1_MNIST_CNN.txt*

#### Mất mát tương phản cho embedding (Contrastive Loss)

- **Là gì:** Hàm loss cho mạng Siamese, nhận nhãn y (0 = cùng loại, 1 = khác loại) và khoảng cách d giữa hai vector embedding. Cặp giống nhau bị phạt theo d^2 (kéo lại gần), cặp khác nhau bị phạt theo max(1-d,0)^2 (đẩy ra xa ít nhất khoảng cách 1).
- **Ý nghĩa & mục đích:** Dùng khi học một HÀM EMBEDDING (one-shot learning) thay vì phân loại trực tiếp. Mục tiêu: đưa ảnh giống nhau lại gần, ảnh khác nhau ra xa tối thiểu 1 đơn vị trong không gian embedding. Thay thế cho log-loss thông thường.
- **Code:**

```python
def contrastive_loss(y, d):
    zero_tensor = torch.zeros(d.shape)
    loss = (((1-y) * torch.square(d))
            + y * torch.square(torch.max(1-d, zero_tensor))).mean()
    return loss
```
- **Visual:** Sau huấn luyện, vẽ histogram chồng: plt.hist(preds[y==0], alpha=0.5, label='same'); plt.hist(preds[y==1], alpha=0.5, label='different'). Hai cụm tách nhau là tốt; còn chồng lấn nhiều (nhất là trên test set) là dấu hiệu overfit.
- **Lưu ý:** Quy ước nhãn ở đây ngược trực giác: y=0 nghĩa GIỐNG nhau, y=1 nghĩa KHÁC nhau. Phần max(1-d,0) tạo 'lề': khi cặp khác đã cách xa >=1 thì không phạt thêm. Nếu hai ảnh trùng khít, d=0 có thể gây chia cho 0 ở nơi khác — dataset loại bỏ cặp trùng index.
- *Nguồn: MNIST_Siamese.txt*

#### Khoảng cách Euclid giữa các vector embedding (Euclidean distance of rows)

- **Là gì:** Hàm tính khoảng cách Euclid theo từng hàng giữa hai batch vector embedding: căn bậc hai của tổng bình phương hiệu.
- **Ý nghĩa & mục đích:** Là 'nguyên liệu' đầu vào cho contrastive loss. Mạng Siamese tính embedding của hai ảnh rồi đo khoảng cách này; contrastive loss dựa trên khoảng cách đó để kéo gần/đẩy xa.
- **Code:**

```python
def euclidean_distances_of_rows(inputs):
    diff = inputs[:,0,:] - inputs[:,1,:]
    return torch.sqrt(torch.sum(torch.square(diff), dim=1, keepdim=True))
```
- **Visual:** Đối chiếu khoảng cách do mạng tính với khoảng cách tính tay trong numpy: plt.plot(distances.reshape([-1]), distances_from_emb01, '.') — phải ra đường thẳng (hàm đồng nhất).
- **Lưu ý:** keepdim=True để giữ shape (n,1) khớp với nhãn y (source viết keepdims cũng chạy vì là alias numpy, nhưng keepdim mới là tên chuẩn của torch). torch.sqrt tại d=0 có gradient vô cực về mặt lý thuyết; tránh cặp ảnh trùng khít. Phải dùng phép tensor của torch (không numpy) để gradient chạy được qua hàm này.
- *Nguồn: MNIST_Siamese.txt*

#### BCELoss trong GAN với nhãn ones/zeros (real vs fake)

- **Là gì:** GAN dùng nn.BCELoss() nhưng nhãn không đến từ dữ liệu mà do ta tự đặt: dữ liệu thật gán nhãn 1 (ones), dữ liệu giả gán nhãn 0 (zeros); generator thì muốn discriminator gán nhãn 1 cho ảnh giả.
- **Ý nghĩa & mục đích:** Đây là cách log-loss được tái sử dụng cho huấn luyện đối kháng. Discriminator học phân biệt thật/giả (loss trên ones cho thật + zeros cho giả). Generator học đánh lừa (loss trên ones cho ảnh giả của mình).
- **Code:**

```python
loss = nn.BCELoss()
ones  = torch.ones(batch_size).to(device)
zeros = torch.zeros(batch_size).to(device)

# Discriminator: that -> 1, gia -> 0
# .view(batch_size) de output (batch,1) khop nhan (batch,)
d_real = loss(discriminator(true_data).view(batch_size), ones)
d_fake = loss(discriminator(generated_data.detach()).view(batch_size), zeros)
discriminator_loss = (d_real + d_fake) / 2

# Generator: muon anh gia bi goi la that -> 1
generator_loss = loss(discriminator(generated_data).view(batch_size), ones)
```
- **Visual:** Vẽ loss_d và loss_g theo epoch (in trong vòng lặp). GAN cân bằng động: hai loss dao động quanh nhau chứ không nhất thiết tụt về 0.
- **Lưu ý:** Khi tính loss cho discriminator trên ảnh giả phải .detach() generated_data để gradient KHÔNG chảy về generator. Nhưng khi tính generator_loss thì KHÔNG detach (cần gradient qua generator). Nhầm chỗ detach là lỗi phổ biến làm GAN không học. Ngoài ra discriminator xuất shape (batch,1) còn nhãn là (batch,): PHẢI .view(batch_size) cho khớp, nếu không BCELoss broadcast âm thầm ra loss sai.
- *Nguồn: CGANs.txt*

#### BCE ổn định số học nhận logits thô (BCEWithLogitsLoss)

- **Là gì:** nn.BCEWithLogitsLoss() gộp sigmoid + BCELoss trong một bước, nhận đầu vào là output THÔ (logit, chưa qua sigmoid) và nhãn 0/1.
- **Ý nghĩa & mục đích:** Đây là cách hiện đại, được khuyến nghị thay cho cặp nn.Sigmoid + nn.BCELoss dùng khắp các notebook. Nó dùng thủ thuật log-sum-exp nên ổn định số học hơn (không bị tràn/underflow khi logit rất lớn hoặc rất nhỏ). Tương đương với CrossEntropyLoss cho bài 2 lớp nhưng chỉ cần 1 output.
- **Code:**

```python
loss_function = nn.BCEWithLogitsLoss()

logits = net(X_tensor)                 # output THO, KHONG co sigmoid o lop cuoi
loss = loss_function(logits, Y_like_output)
loss.backward()

# Khi can xac suat de suy luan:
probs = torch.sigmoid(logits)
```
- **Visual:** Loss chưa học vẫn xấp xỉ 0.69 như BCELoss (vì cùng là log-loss 2 lớp). Vẽ train/val loss theo epoch y hệt.
- **Lưu ý:** Nếu chuyển sang BCEWithLogitsLoss thì phải BỎ nn.Sigmoid ở lớp cuối của mạng — để cả sigmoid rồi lại đưa vào đây là sigmoid HAI lần, sai hoàn toàn. Muốn ra xác suất lúc inference phải tự torch.sigmoid(logits). Có tham số pos_weight để xử lý lớp mất cân bằng.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch.txt; Week3Exercise2_Percolation_CNN.txt; CGANs.txt*

#### Tham số reduction của hàm loss (mean/sum/none)

- **Là gì:** Mọi hàm loss của PyTorch có tham số reduction quyết định cách gộp loss từng ví dụ: 'mean' (mặc định, trung bình), 'sum' (tổng), hoặc 'none' (trả về vector loss từng ví dụ, không gộp).
- **Ý nghĩa & mục đích:** Mặc định 'mean' cho ra một số không phụ thuộc kích thước batch, dễ so sánh. 'none' hữu ích khi muốn cân trọng số từng ví dụ, che (mask) một số ví dụ, hoặc phân tích xem ví dụ nào khó nhất. 'sum' dùng khi tự cộng dồn thủ công.
- **Code:**

```python
# mac dinh: tra ve 1 so
loss_mean = nn.BCELoss()(output, target)

# 'none': tra ve loss tung vi du (cung shape target)
loss_none = nn.BCELoss(reduction='none')(output, target)
per_example = loss_none            # co the nhan trong so roi .mean()
weighted = (per_example * w).mean()
```
- **Visual:** Vẽ histogram per_example để xem phân bố độ khó: plt.hist(per_example.detach().numpy()) — đuôi phải là các ví dụ mạng sai nặng.
- **Lưu ý:** Mặc định là 'mean'; đổi sang 'sum' thì loss phụ thuộc batch size nên phải chỉnh lại learning rate. reduction='none' trả tensor nhiều phần tử, KHÔNG gọi .backward() trực tiếp được — phải reduce (mean/sum) trước.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt; Week3Exercise2_Percolation_CNN.txt*

#### NLLLoss + log_softmax (thành phần của CrossEntropy)

- **Là gì:** nn.CrossEntropyLoss thực chất = nn.LogSoftmax + nn.NLLLoss. NLLLoss (negative log-likelihood) nhận đầu vào là log-xác suất (đã qua log_softmax) và nhãn lớp nguyên, rồi lấy âm log-xác suất của lớp đúng.
- **Ý nghĩa & mục đích:** Hiểu sự phân rã này giúp: (1) thấy CrossEntropyLoss không có gì huyền bí, chỉ là log-loss thủ công ở thẻ trước; (2) linh hoạt khi cần chính log-xác suất ở lớp cuối (vd một số kiến trúc muốn xuất log-prob). Nếu mạng đã có F.log_softmax ở cuối thì dùng NLLLoss (đừng dùng CrossEntropyLoss nữa).
- **Code:**

```python
import torch.nn.functional as F

log_probs = F.log_softmax(logits, dim=1)   # output tho -> log xac suat
loss = nn.NLLLoss()(log_probs, labels)     # labels la LongTensor chi so lop

# tuong duong:
loss2 = nn.CrossEntropyLoss()(logits, labels)   # loss == loss2
```
- **Visual:** Kiểm tra bằng đẳng thức: NLLLoss(log_softmax(logits)) phải bằng CrossEntropyLoss(logits) trên cùng batch — in hai số ra so sánh.
- **Lưu ý:** NLLLoss cần đầu vào là LOG-xác suất (qua log_softmax), KHÔNG phải xác suất thô hay logit thô. Đừng ghép log_softmax rồi lại đưa vào CrossEntropyLoss (thành softmax/log hai lần). Nhãn là chỉ số lớp nguyên, không one-hot.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*


---

### 8.5 Pipeline dữ liệu (Dataset/DataLoader/transforms)

#### Gộp chuỗi biến đổi (transforms.Compose)

- **Là gì:** Đối tượng gói nhiều phép biến đổi (transform) lại thành một pipeline chạy tuần tự trên mỗi mẫu.
- **Ý nghĩa & mục đích:** Dùng để khai báo một lần toàn bộ chuỗi tiền xử lý ảnh (đổi sang tensor, chuẩn hoá, cắt, lật...) rồi gắn vào Dataset qua tham số transform. Mỗi ảnh khi lấy ra sẽ tự động đi qua đúng thứ tự này.
- **Code:**

```python
transform_list = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0], std=[1.0])
])
```
- **Visual:** Hình dung như một dây chuyền: ảnh gốc -> ToTensor -> Normalize -> tensor sẵn sàng. Không có plot riêng; có thể kiểm tra bằng cách in shape/giá trị của một mẫu sau khi qua Compose.
- **Lưu ý:** Thứ tự quan trọng: ToTensor phải đứng trước Normalize (Normalize cần tensor, không nhận PIL Image). Transform augmentation ngẫu nhiên chỉ nên đặt ở tập train, không đặt ở val.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Đổi ảnh sang tensor (transforms.ToTensor)

- **Là gì:** Phép biến đổi chuyển ảnh PIL/numpy (H,W,C) thành torch tensor (C,H,W) và đưa giá trị pixel từ [0,255] về [0,1].
- **Ý nghĩa & mục đích:** Bước bắt buộc đầu tiên của pipeline ảnh: mạng nơ-ron làm việc với tensor, và việc đưa pixel về [0,1] đã là một dạng chuẩn hoá cơ bản giúp học ổn định hơn.
- **Code:**

```python
transforms.ToTensor()
```
- **Visual:** So sánh trước/sau: ảnh gốc pixel 0-255, sau ToTensor giá trị 0.0-1.0 và trục kênh chuyển lên đầu (1,28,28 cho ảnh xám). Kiểm tra bằng mnist_trainset[0][0].shape.
- **Lưu ý:** ToTensor tự chia 255, nên nếu sau đó Normalize với mean=0,std=1 thì phép tính (x-0)/1 giữ nguyên giá trị trong [0,1] chứ không thành mean 0/std 1 thật (đúng điểm gây bối rối trong notebook, không phải lỗi).
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Chuẩn hoá theo mean/std (transforms.Normalize)

- **Là gì:** Phép biến đổi lấy tensor trừ đi mean rồi chia cho std, tính theo từng kênh: output = (input - mean) / std.
- **Ý nghĩa & mục đích:** Đưa các đầu vào về cùng khoảng độ lớn typical. Quan trọng vì gradient của một trọng số tỉ lệ với đầu vào; nếu các đầu vào chênh lệch độ lớn lớn thì việc tối ưu khó khăn. Ảnh ImageNet dùng mean/std chuẩn theo 3 kênh RGB.
- **Code:**

```python
# anh xam MNIST
transforms.Normalize(mean=[0.0], std=[1.0])
# anh mau ImageNet (3 kenh)
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```
- **Visual:** Histogram giá trị pixel trước và sau Normalize: sau chuẩn hoá phân bố nên tập trung quanh 0 với độ lớn hợp lý (vd -2 đến +4). plt.hist(x_train.reshape(-1).numpy()).
- **Lưu ý:** mean/std là danh sách theo số kênh: ảnh xám 1 phần tử, ảnh màu 3 phần tử. Phải dùng ĐÚNG mean/std của tập pretrain khi transfer learning, nếu không đặc trưng sẽ lệch.
- *Nguồn: transfer_learning_tutorial.txt*

#### Bộ dữ liệu dựng sẵn của torchvision (datasets.MNIST)

- **Là gì:** Lớp Dataset có sẵn, tự tải MNIST về đĩa và trả về từng mẫu là cặp (ảnh, nhãn) đã qua transform.
- **Ý nghĩa & mục đích:** Cho phép lấy ngay một dataset chuẩn để thử nghiệm mà không phải tự đọc file. root chỉ thư mục lưu, train chọn tập train/test, download=True tự tải nếu chưa có, transform gắn pipeline tiền xử lý.
- **Code:**

```python
mnist_trainset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform_list)
mnist_testset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform_list)
```
- **Visual:** plt.imshow(mnist_trainset[0][0].view(28,28).numpy()); plt.colorbar() để xem một chữ số. In mnist_trainset để thấy summary (số mẫu, transform).
- **Lưu ý:** fashion_MNIST là bản thay thế cùng API nhưng tính chất khác. type(mnist_trainset) không cho thông tin hữu ích; dùng len() để biết số mẫu (60000).
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Cấu trúc Dataset: cặp (ảnh, nhãn) và lập chỉ mục

- **Là gì:** Một Dataset hành xử như danh sách: dataset[i] trả về một cặp, dataset[i][0] là ảnh (tensor), dataset[i][1] là nhãn; len(dataset) cho số mẫu.
- **Ý nghĩa & mục đích:** Đây là giao diện tối thiểu mà DataLoader dựa vào để rút mẫu. Hiểu cấu trúc này giúp truy cập, cắt lát và trực quan hoá dữ liệu trước khi huấn luyện.
- **Code:**

```python
len(mnist_trainset)            # 60000
im0 = mnist_trainset[0][0]     # tensor (1,28,28)
label0 = mnist_trainset[0][1]  # so nguyen
```
- **Visual:** Vẽ lưới ảnh kèm nhãn: lấy [dataset[i][0] for i in range(36)] và [dataset[i][1] ...] rồi imshow từng ô với set_title(nhãn).
- **Lưu ý:** Ảnh MNIST là (1,28,28); để đưa vào matplotlib phải reshape(28,28).numpy(). Để đưa vào model conv cần thêm chiều batch: .view(1,1,28,28).
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Lấy tập con nhanh bằng list comprehension

- **Là gì:** Tạo một danh sách các cặp (ảnh,nhãn) từ một dải chỉ mục của dataset gốc để dùng như một dataset nhỏ hơn.
- **Ý nghĩa & mục đích:** Khi muốn thí nghiệm nhanh (đổi kiến trúc, tham số) nên train trên tập nhỏ để mỗi vòng chạy nhanh, học được nhiều hơn từ nhiều lần thử. DataLoader nhận thẳng danh sách này.
- **Code:**

```python
mnist_trainset_small = [ mnist_trainset[i] for i in range(0, 4000) ]
```
- **Visual:** Không có hình riêng; kiểm tra bằng len(mnist_trainset_small) và vẽ vài mẫu đầu để chắc dữ liệu đúng.
- **Lưu ý:** Danh sách Python này giữ các mẫu trong RAM; với dataset lớn sẽ tốn bộ nhớ. Đây là cách nhanh-gọn cho tập nhỏ, không phải cách chuẩn cho tập lớn (nên dùng Subset/random_split).
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Truy cập dữ liệu thô .data và nhãn .targets

- **Là gì:** Thuộc tính của dataset torchvision cho phép lấy nguyên khối tensor ảnh (.data, kiểu uint8) và tensor nhãn (.targets), bỏ qua transform.
- **Ý nghĩa & mục đích:** Dùng khi cần thao tác toàn bộ dữ liệu ở mức tensor: tính mean ảnh, độ lệch chuẩn từng pixel, tự chuẩn hoá thủ công thay vì qua transform pipeline.
- **Code:**

```python
mnist_trainset.data.shape      # (60000,28,28)
mnist_trainset.targets         # tensor nhan
mean_image = torch.mean(mnist_trainset.data.type(torch.DoubleTensor), axis=0)
```
- **Visual:** plt.imshow(mean_image); plt.colorbar() cho thấy ảnh trung bình (pixel giữa quan trọng hơn). plt.hist(pixel_std.numpy()) xem phân bố std từng pixel.
- **Lưu ý:** .data là uint8 (0-255) CHƯA qua transform — phải .type(torch.DoubleTensor) trước khi tính mean/std, nếu không sẽ tràn/sai kiểu. Khác hoàn toàn với dataset[i][0] (đã qua transform).
- *Nguồn: MNIST_Siamese.txt*

#### Bộ nạp minibatch (DataLoader)

- **Là gì:** Lớp tiện ích bọc quanh một Dataset, cung cấp khả năng lặp để rút ra từng minibatch với kích thước batch_size, có thể xáo trộn.
- **Ý nghĩa & mục đích:** Chia dữ liệu lớn thành các batch nhỏ cho SGD, tự lo việc gộp mẫu thành tensor batch và (tuỳ chọn) trộn thứ tự mỗi epoch. Là mắt xích trung tâm của vòng huấn luyện.
- **Code:**

```python
trainloader = torch.utils.data.DataLoader(mnist_trainset_small, batch_size=32, shuffle=True)
testloader  = torch.utils.data.DataLoader(mnist_testset,       batch_size=32, shuffle=False)
```
- **Visual:** Không vẽ trực tiếp; hình dung dataset bị cắt thành các khối batch_size mẫu. Có thể lấy một batch ra kiểm tra shape (batch, C, H, W).
- **Lưu ý:** DataLoader nhẹ, chỉ cung cấp iteration — muốn đổi batch_size phải tạo loader mới. shuffle=True cho train (bắt buộc để SGD tốt); tập test/val nên để shuffle=False (nhất quán, tiết kiệm).
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Số tiến trình nạp dữ liệu (num_workers)

- **Là gì:** Tham số của DataLoader quy định số tiến trình con chạy song song để nạp và tiền xử lý dữ liệu.
- **Ý nghĩa & mục đích:** Tăng num_workers để việc đọc/biến đổi ảnh chạy song song với việc GPU tính toán, tránh nghẽn cổ chai I/O khi dataset đọc từ đĩa (vd ImageFolder).
- **Code:**

```python
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
```
- **Visual:** Không có hình; hiệu quả thể hiện gián tiếp qua thời gian mỗi epoch giảm.
- **Lưu ý:** Trên Windows/notebook, num_workers>0 đôi khi gây lỗi do cơ chế multiprocessing (spawn); nếu gặp lỗi hãy đặt num_workers=0, hoặc bọc code chạy trong if __name__=='__main__'. Có thể thêm pin_memory=True khi dùng GPU.
- *Nguồn: transfer_learning_tutorial.txt*

#### Lấy một minibatch để kiểm tra (iter + next)

- **Là gì:** Tạo một iterator tạm từ DataLoader rồi gọi next để lấy một batch duy nhất gồm (ảnh, nhãn).
- **Ý nghĩa & mục đích:** Cách nhanh để soi một batch trước khi vào vòng train: kiểm tra shape, kiểu, giá trị nhãn, hoặc để trực quan hoá vài ảnh.
- **Code:**

```python
tmpiter = iter(trainloader)
images, labels = next(tmpiter)   # cach nhanh: next(iter(trainloader))
images.shape                     # (batch,1,28,28)
```
- **Visual:** Sau khi lấy batch, dùng make_grid + imshow để xem nguyên batch trong một hình.
- **Lưu ý:** API cũ tmpiter.next() ĐÃ LỖI THỜI (bị bỏ ở PyTorch mới) — phải dùng next(tmpiter) hoặc next(iter(loader)). Batch trả về là list [images, labels].
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Lặp qua DataLoader trong vòng huấn luyện

- **Là gì:** Vòng for lấy lần lượt từng minibatch từ loader; mỗi mẫu lặp trả về cặp (inputs, labels).
- **Ý nghĩa & mục đích:** Đây là khung xương của một epoch: duyệt hết mọi batch, mỗi batch làm forward + loss + backward + step. enumerate cho thêm chỉ số batch.
- **Code:**

```python
for i, mini_batch in enumerate(trainloader, 0):
    images, labels = mini_batch
    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
```
- **Visual:** Đường loss theo batch/epoch: cộng dồn loss.item() rồi chia số batch, vẽ theo epoch.
- **Lưu ý:** Nhớ optimizer.zero_grad() mỗi batch (nếu quên, gradient cộng dồn gây sai). loss là tensor 0 chiều — lấy giá trị bằng loss.item(), không index được. Cộng dồn nên dùng loss.item() (không giữ cả graph gây rò bộ nhớ).
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*

#### Dataset theo thư mục nhãn (datasets.ImageFolder)

- **Là gì:** Lớp Dataset tự động đọc ảnh từ cây thư mục, trong đó mỗi thư mục con là một lớp; tên thư mục trở thành nhãn.
- **Ý nghĩa & mục đích:** Cách chuẩn để dùng dữ liệu ảnh của riêng mình mà không cần viết code đọc file: chỉ cần sắp ảnh vào các thư mục theo lớp (vd train/ants, train/bees). Trả về classes tự động.
- **Code:**

```python
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
class_names   = image_datasets['train'].classes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
```
- **Visual:** Lấy một batch và make_grid để xem ảnh cùng nhãn class_names — kiểm tra thư mục đã gán đúng lớp chưa.
- **Lưu ý:** Cấu trúc thư mục phải đúng: gốc/lớp_A/*.jpg, gốc/lớp_B/*.jpg. Thứ tự lớp theo thứ tự alphabet của tên thư mục, ảnh hưởng chỉ số nhãn (class_to_idx).
- *Nguồn: transfer_learning_tutorial.txt*

#### Tăng cường dữ liệu ảnh (augmentation transforms)

- **Là gì:** Các phép biến đổi ngẫu nhiên/cố định làm giàu và chuẩn kích thước ảnh: RandomResizedCrop, RandomHorizontalFlip (train); Resize + CenterCrop (val).
- **Ý nghĩa & mục đích:** Ở tập train, biến đổi ngẫu nhiên tạo thêm biến thể để mạng khái quát tốt hơn, chống overfit. Ở tập val chỉ resize/crop cố định để đánh giá nhất quán, không thêm ngẫu nhiên.
- **Code:**

```python
'train': transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
'val': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
```
- **Visual:** make_grid một batch train để tận mắt thấy ảnh bị crop/lật khác nhau mỗi lần; so với batch val thì ổn định.
- **Lưu ý:** TUYỆT ĐỐI không đặt augmentation ngẫu nhiên ở tập val/test (làm kết quả đánh giá không lặp lại được). Kích thước crop (224) phải khớp đầu vào mạng pretrain.
- *Nguồn: transfer_learning_tutorial.txt*

#### Tổ chức pipeline theo pha bằng dict (train/val)

- **Là gì:** Idiom dùng dictionary khoá theo pha 'train'/'val' cho transforms, datasets, dataloaders và kích thước, để code huấn luyện chọn pha bằng một biến.
- **Ý nghĩa & mục đích:** Giúp một hàm train_model duy nhất chạy được cả pha train và val chỉ bằng dataloaders[phase], tránh lặp code và giảm nhầm lẫn giữa hai pha.
- **Code:**

```python
data_transforms = {'train': ..., 'val': ...}
image_datasets  = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train','val']}
dataloaders     = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True) for x in ['train','val']}
for phase in ['train', 'val']:
    for inputs, labels in dataloaders[phase]: ...
```
- **Visual:** Không có hình; giá trị nằm ở tính gọn gàng của vòng lặp huấn luyện hai pha.
- **Lưu ý:** Phải gọi model.train() ở pha train và model.eval() ở pha val để dropout/batchnorm hoạt động đúng chế độ.
- *Nguồn: transfer_learning_tutorial.txt*

#### Ghép batch thành lưới ảnh (make_grid)

- **Là gì:** Hàm torchvision.utils.make_grid gộp một batch tensor ảnh thành một tensor ảnh lưới duy nhất để hiển thị.
- **Ý nghĩa & mục đích:** Trực quan hoá nhanh cả một batch trong một khung hình, kiểm tra augmentation và nhãn có hợp lý không trước khi train.
- **Code:**

```python
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])
```
- **Visual:** Chính nó là một hình: một lưới các ảnh trong batch, kèm tiêu đề là danh sách tên lớp tương ứng.
- **Lưu ý:** Ảnh trong batch đã bị Normalize nên cần giải chuẩn hoá trước khi imshow, nếu không màu sẽ sai/lệch.
- *Nguồn: transfer_learning_tutorial.txt*

#### Giải chuẩn hoá để hiển thị ảnh (denormalize imshow)

- **Là gì:** Hàm đảo ngược Normalize trước khi vẽ: đổi trục về (H,W,C), nhân lại std, cộng lại mean, rồi clip về [0,1].
- **Ý nghĩa & mục đích:** Tensor ảnh sau Normalize có giá trị âm/dương lệch nên nếu vẽ thẳng sẽ sai màu. Phải khôi phục về khoảng [0,1] mới hiển thị đúng như mắt người thấy.
- **Code:**

```python
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
```
- **Visual:** Chính là bước để plt.imshow ra ảnh màu đúng; so sánh với vẽ tensor chưa giải chuẩn (bị ám màu) để thấy khác biệt.
- **Lưu ý:** Phải dùng ĐÚNG mean/std đã dùng khi Normalize. transpose((1,2,0)) chuyển từ (C,H,W) của torch sang (H,W,C) mà matplotlib cần. Nếu tensor ở GPU phải .cpu() trước khi .numpy().
- *Nguồn: transfer_learning_tutorial.txt*

#### Dataset từ tensor có sẵn (TensorDataset)

- **Là gì:** Lớp bọc một hoặc nhiều tensor cùng chiều đầu tiên thành một Dataset, mỗi phần tử là một lát cắt tương ứng của các tensor (vd (x_i, y_i)).
- **Ý nghĩa & mục đích:** Khi dữ liệu đã ở dạng tensor trong bộ nhớ (đã tiền xử lý, đã tokenize) thì đây là cách nhanh nhất để đưa vào DataLoader mà không cần viết lớp Dataset riêng.
- **Code:**

```python
train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
test_data  = TensorDataset(torch.LongTensor(x_val),   torch.LongTensor(y_val))
train_loader = DataLoader(train_data, shuffle=True, batch_size=128)
```
- **Visual:** Không có hình; kiểm tra bằng cách lấy một batch và in shape (batch, seq_len) cho x, (batch,) cho y.
- **Lưu ý:** Các tensor phải cùng kích thước chiều 0 (số mẫu). Kiểu phải khớp yêu cầu model: chỉ số từ vựng / nhãn phân loại thường là int64. torch.LongTensor(...) là constructor CŨ — bản mới nên dùng x.long() hoặc torch.as_tensor(x, dtype=torch.long) để không copy thừa.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Chuyển batch sang thiết bị (.to(device), non_blocking)

- **Là gì:** Phương thức chuyển tensor (ảnh/nhãn) sang GPU/CPU đã chọn; non_blocking=True cho phép copy bất đồng bộ.
- **Ý nghĩa & mục đích:** Model và dữ liệu phải cùng một thiết bị mới tính được. Trong vòng train, mỗi batch lấy từ loader (ở CPU) cần đẩy sang device trước khi forward.
- **Code:**

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for inputs, targets in train_loader:
    inputs  = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
```
- **Visual:** Không có hình; lỗi thường gặp 'expected all tensors on same device' báo hiệu quên .to(device).
- **Lưu ý:** Phải chuyển CẢ model (model.to(device)) lẫn từng batch. Khi lấy kết quả về để vẽ/numpy phải .cpu().detach().numpy(). non_blocking chỉ có tác dụng khi DataLoader dùng pin_memory=True và chuyển sang CUDA.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Tách token cho văn bản (get_tokenizer)

- **Là gì:** Hàm của torchtext trả về một bộ tách từ; 'basic_english' hạ chữ thường và tách theo dấu cách/dấu câu thành danh sách token.
- **Ý nghĩa & mục đích:** Bước đầu của pipeline NLP: biến câu văn thô thành danh sách các từ/token, làm nền cho việc xây từ vựng và mã hoá.
- **Code:**

```python
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')
tokens = tokenizer('You can now install TorchText using pip!')
# ['you','can','now','install','torchtext','using','pip','!']
```
- **Visual:** Không có hình; in trực tiếp danh sách token để kiểm tra cách tách.
- **Lưu ý:** torchtext ĐÃ NGỪNG PHÁT TRIỂN và bị gỡ khỏi các bản PyTorch mới nhất — API có thể không cài/không chạy được. Với dự án mới nên cân nhắc tokenizer khác (spaCy, HuggingFace tokenizers). Cần cài đúng phiên bản torchtext tương thích với phiên bản torch.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Xây từ vựng từ dữ liệu (build_vocab_from_iterator)

- **Là gì:** Hàm duyệt qua các chuỗi token để dựng từ điển ánh xạ từ -> chỉ số nguyên; min_freq bỏ từ hiếm, specials thêm token đặc biệt như <unk>.
- **Ý nghĩa & mục đích:** Model chỉ nhận số, không nhận chữ. Vocab biến mỗi từ thành một chỉ số. Bỏ từ hiếm (min_freq) làm nhỏ từ vựng, giảm nhiễu; <unk> gom mọi từ ngoài từ vựng.
- **Code:**

```python
from torchtext.vocab import build_vocab_from_iterator
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)
vocab = build_vocab_from_iterator(yield_tokens(iter(texts)), min_freq=30, specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])
```
- **Visual:** len(vocab) cho kích thước từ vựng. Có thể vẽ histogram tần suất từ để chọn ngưỡng min_freq.
- **Lưu ý:** BẮT BUỘC set_default_index(vocab['<unk>']) — nếu không, gặp từ lạ (ngoài từ vựng) sẽ báo lỗi thay vì trả về <unk>. Chỉ được xây vocab từ tập TRAIN để tránh rò rỉ thông tin. (torchtext.vocab cũng thuộc API đã ngừng phát triển.)
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Mã hoá câu thành chuỗi chỉ số (vocab(tokens))

- **Là gì:** Gọi vocab lên danh sách token để nhận về danh sách chỉ số nguyên tương ứng của từng từ.
- **Ý nghĩa & mục đích:** Biến một review văn bản thành một tensor chỉ số mà lớp Embedding có thể tra cứu. Kết hợp tokenizer + vocab thành một bước mã hoá gọn.
- **Code:**

```python
seq = torch.tensor(vocab(tokenizer(text)[0:maxlen]))
# vd: vocab(tokenizer(texts[1]))
```
- **Visual:** In song song text[i] và vocab(tokenizer(text[i])) để đối chiếu từ với chỉ số.
- **Lưu ý:** Cắt [0:maxlen] để giới hạn độ dài; câu dài hơn bị chặt. Phải dùng CÙNG tokenizer và vocab lúc train khi encode dữ liệu mới (vd encode_my_reviews truy cập tokenizer như biến toàn cục).
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Đệm chuỗi cho bằng độ dài (pad_sequence)

- **Là gì:** Hàm gộp một danh sách tensor 1 chiều có độ dài khác nhau thành một tensor 2 chiều bằng cách chèn giá trị đệm (padding_value) cho đủ độ dài lớn nhất.
- **Ý nghĩa & mục đích:** Batch trong RNN phải là tensor chữ nhật, nhưng các câu dài ngắn khác nhau. pad_sequence đệm để mọi câu cùng độ dài, gộp được thành ma trận (số_câu, độ_dài).
- **Code:**

```python
from torch.nn.utils.rnn import pad_sequence
sequences = [torch.tensor(vocab(tokenizer(t)[0:maxlen])) for t in texts]
data = pad_sequence(sequences, batch_first=True, padding_value=0)
print('Shape of data tensor:', data.shape)
```
- **Visual:** In data.shape thấy (n_texts, max_len_thuc_te). Có thể imshow một phần ma trận để thấy vùng đệm 0 ở cuối các câu ngắn.
- **Lưu ý:** Tham số thứ 2 (batch_first) đặt True để chiều batch đứng đầu (source viết dạng vị trí: pad_sequence(seq, True, ...)). pad_sequence đệm tới độ dài LỚN NHẤT trong danh sách, không tự cắt tới maxlen cố định — nếu cần chiều cố định phải tự cắt/đệm thêm. padding_value=0 nên chỉ số 0 dành riêng cho token đệm.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Đọc dữ liệu thô từ cây thư mục

- **Là gì:** Idiom duyệt thủ công các thư mục lớp, đọc từng file .txt và gán nhãn theo tên thư mục, dựng hai danh sách texts và labels.
- **Ý nghĩa & mục đích:** Khi dữ liệu là văn bản/tệp tự tổ chức (vd aclImdb với neg/pos), cần tự đọc từ đĩa thành list trong bộ nhớ trước khi tokenize và mã hoá.
- **Code:**

```python
import os
train_dir = os.path.join(imdb_dir, 'train')
texts, labels = [], []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname.endswith('.txt'):
            with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                texts.append(f.read())
            labels.append(0 if label_type == 'neg' else 1)
```
- **Visual:** plt.hist(labels) để xem phân bố nhãn (cân bằng hay không).
- **Lưu ý:** Nên mở file với encoding='utf-8' (mặc định của open() khác nhau theo hệ điều hành — dễ lỗi UnicodeDecodeError trên Windows). Dùng with open(...) để tự đóng file. Đọc hàng chục nghìn file chậm — nên chạy một lần rồi lưu ra pickle; in tiến độ mỗi 1000 file để biết còn chạy.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Lưu/nạp dữ liệu tiền xử lý (pickle)

- **Là gì:** pickle là bộ tuần tự hoá của Python; dump ghi các biến ra file nhị phân, load nạp lại y nguyên.
- **Ý nghĩa & mục đích:** Tiền xử lý (đọc file, tokenize, dựng vocab) tốn thời gian; lưu kết quả ra pickle để lần sau nạp thẳng, bỏ qua bước chậm.
- **Code:**

```python
import pickle
with open('imdb_raw_and_coded_data.pickle', 'wb') as fh:
    pickle.dump((texts, labels, vocab, x_train, y_train, x_val, y_val, tokenizer), fh)
# lan sau:
with open('imdb_raw_and_coded_data.pickle', 'rb') as fh:
    (texts, labels, vocab, x_train, y_train, x_val, y_val, tokenizer) = pickle.load(fh)
```
- **Visual:** Không có hình; kiểm tra bằng cách in shape các biến sau khi load.
- **Lưu ý:** File pickle có thể KHÔNG nạp được nếu đổi phiên bản Python/thư viện. Chỉ chạy dump khi vừa xử lý xong dữ liệu mới (dòng cảnh báo 'DO NOT RUN this unless...'). KHÔNG mở pickle từ nguồn không tin cậy (có thể chạy mã độc khi load).
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Chia train/val bằng cắt lát (slicing)

- **Là gì:** Tách dữ liệu đã gộp thành tập train và validation bằng cách cắt theo chỉ số trên tensor/mảng.
- **Ý nghĩa & mục đích:** Cách đơn giản để tạo tập kiểm định khi dữ liệu đã ở dạng tensor phẳng: n mẫu đầu làm train, phần còn lại làm val.
- **Code:**

```python
training_samples   = 20000
validation_samples = 5000
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val   = data[training_samples: training_samples + validation_samples]
y_val   = labels[training_samples: training_samples + validation_samples]
```
- **Visual:** plt.hist(y_train.numpy()) và plt.hist(y_val.numpy()) để kiểm tra nhãn hai tập có cân bằng không.
- **Lưu ý:** Nếu dữ liệu chưa được trộn trước khi cắt, tập train/val có thể lệch phân bố nhãn (như bài IMDB: neg đứng trước, pos đứng sau nên cắt thẳng làm hai tập rất lệch). Nên shuffle trước hoặc dùng random_split, và luôn kiểm tra histogram.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Chuẩn hoá thủ công bằng broadcasting

- **Là gì:** Tự chuẩn hoá dữ liệu ở mức tensor: trừ ảnh trung bình (mean image) khỏi mọi ảnh nhờ cơ chế broadcasting của torch.
- **Ý nghĩa & mục đích:** Thay cho transforms.Normalize, đôi khi ta tự tính mean/std từ dữ liệu và trừ đi. Broadcasting cho phép trừ một tensor nhỏ hơn (một ảnh trung bình) khỏi cả khối nhiều ảnh mà không cần vòng lặp.
- **Code:**

```python
mean_image = torch.mean(mnist_trainset.data.type(torch.DoubleTensor), dim=0)
x_train = mnist_trainset.data.type(torch.DoubleTensor) - mean_image
x_test  = mnist_testset.data.type(torch.DoubleTensor)  - mean_image
```
- **Visual:** plt.imshow(mean_image); plt.colorbar() để thấy ảnh trung bình bị trừ đi. Histogram giá trị sau khi trừ để xem đã tâm quanh 0 chưa.
- **Lưu ý:** Mean của tập test nên tính từ THỐNG KÊ TRAIN để tránh rò rỉ (notebook gốc tính riêng cho tiện — đây là điểm cần chú ý). Phải .type(torch.DoubleTensor) trước vì .data là uint8. Không nên tái định nghĩa x_train nhiều lần (mất bản gốc).
- *Nguồn: MNIST_Siamese.txt*

#### Chuẩn hoá bằng độ lệch chuẩn toàn cục

- **Là gì:** Chia toàn bộ giá trị pixel cho một độ lệch chuẩn duy nhất tính trên tất cả pixel, thay vì std riêng từng pixel.
- **Ý nghĩa & mục đích:** Với MNIST nhiều pixel gần như luôn bằng 0; chia theo std từng pixel sẽ thổi phồng các pixel hiếm. Chia theo std chung đưa mọi giá trị về khoảng nhỏ hợp lý mà không khuếch đại nhiễu.
- **Code:**

```python
std_train = torch.std(x_train.reshape(-1))
x_train = x_train / std_train
x_test  = x_test / std_train
```
- **Visual:** plt.hist(x_train.reshape(-1).numpy()) — mong đợi phần lớn giá trị nằm trong khoảng nhỏ (vd -2 đến +4).
- **Lưu ý:** Dùng std của TRAIN cho cả test (nhất quán, tránh rò rỉ). Giá trị đầu vào quá lớn có thể làm bão hoà nơ-ron tanh, gradient nhỏ, học chậm ở giai đoạn đầu.
- *Nguồn: MNIST_Siamese.txt*

#### Dựng dataset cặp tuỳ biến rồi stack/permute

- **Là gì:** Tự viết hàm tạo dữ liệu gồm các cặp mẫu và nhãn quan hệ, sau đó dùng torch.stack và .permute để sắp lại các chiều cho đúng định dạng model.
- **Ý nghĩa & mục đích:** Với bài toán học embedding (Siamese), mỗi mẫu là MỘT CẶP ảnh và nhãn giống/khác. Cần tự dựng dữ liệu cặp rồi gộp thành tensor và hoán vị chiều để chiều kênh về đúng vị trí.
- **Code:**

```python
pairs_train, py_train = construct_pairs_dataset(2000, x_train, y_train)
pairs_train_tensor = torch.stack(pairs_train)              # (2, N, 28,28,1)
pairs_train_tensor = pairs_train_tensor.permute(1,0,4,2,3) # (N, 2, 1,28,28)
dataset = torch.utils.data.TensorDataset(pairs_train_tensor, py_train)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
```
- **Visual:** Vẽ hai ảnh của một cặp cạnh nhau (subplots 1x2) kèm nhãn y để xác nhận: cùng chữ số -> y=0, khác -> y=1.
- **Lưu ý:** Rất dễ sai thứ tự chiều — luôn in .shape sau mỗi stack/permute để kiểm tra. Ở đây permute(1,0,4,2,3) đưa channel về chiều thứ 3 để có (N,2,C,H,W) mà conv2d cần khi tách từng ảnh của cặp.
- *Nguồn: MNIST_Siamese.txt*

#### Tự viết lớp Dataset (subclass torch.utils.data.Dataset)

- **Là gì:** Cách chuẩn nhất để định nghĩa dữ liệu riêng: kế thừa Dataset và cài đặt hai phương thức __len__ (số mẫu) và __getitem__(idx) (trả về một mẫu, thường là cặp (x, y)).
- **Ý nghĩa & mục đích:** Khi dữ liệu không vừa dạng dựng sẵn (TensorDataset/ImageFolder) — vd đọc lazy từng file lớn từ đĩa, ghép nhiều nguồn, áp transform tuỳ biến theo mẫu — thì tự viết Dataset là cách tái sử dụng, sạch và tiết kiệm RAM. DataLoader làm việc với BẤT KỲ đối tượng nào có __len__ và __getitem__.
- **Code:**

```python
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        x, y = self.items[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
```
- **Visual:** Không có hình; kiểm tra bằng len(ds) và ds[0] rồi in shape/kiểu mẫu trả về trước khi bọc vào DataLoader.
- **Lưu ý:** __getitem__ phải trả về tensor (hoặc thứ collate được), không trả về PIL/đường dẫn thô. Muốn đọc lazy thì mở/đọc file BÊN TRONG __getitem__ (không nạp hết vào RAM trong __init__). Đây chính là giao diện mà list-comprehension, TensorDataset và ImageFolder đều tuân theo.
- *Nguồn: (bổ sung — chuẩn PyTorch, không có trong notebook)*

#### Chia train/val ngẫu nhiên (random_split)

- **Là gì:** Hàm torch.utils.data.random_split cắt một Dataset thành các Dataset con không giao nhau với kích thước cho trước, chọn mẫu ngẫu nhiên.
- **Ý nghĩa & mục đích:** An toàn hơn cắt lát theo chỉ số: tự trộn nên tránh lệch phân bố nhãn khi dữ liệu chưa shuffle (đúng vấn đề của bài IMDB). Đặt generator có seed để lần chạy nào cũng ra cùng một cách chia (tái lập được).
- **Code:**

```python
from torch.utils.data import random_split
n_val = 5000
n_train = len(dataset) - n_val
train_set, val_set = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42))
```
- **Visual:** Không có hình; sau khi chia, vẽ histogram nhãn của train_set và val_set (duyệt lấy y) để xác nhận hai tập cân bằng.
- **Lưu ý:** Chỉ chia MỘT LẦN và cố định seed, nếu không mỗi lần chạy lại trộn khác nhau làm rò rỉ giữa các thí nghiệm. Tổng các phần phải bằng len(dataset). random_split trả về đối tượng Subset (vẫn tham chiếu dataset gốc, không copy dữ liệu).
- *Nguồn: (bổ sung — chuẩn PyTorch, không có trong notebook)*

#### Tắt gradient khi duyệt loader để đánh giá (torch.no_grad + eval)

- **Là gì:** Khi lặp DataLoader để tính accuracy/loss trên tập val/test, bọc vòng lặp trong with torch.no_grad() và đặt model.eval().
- **Ý nghĩa & mục đích:** Ở pha đánh giá không cần gradient: no_grad tắt việc dựng đồ thị tính đạo hàm, tiết kiệm bộ nhớ và chạy nhanh hơn; model.eval() chuyển dropout/batchnorm sang chế độ suy luận để kết quả nhất quán, không ngẫu nhiên.
- **Code:**

```python
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
```
- **Visual:** Không có hình; in tỉ lệ đúng correct/total sau mỗi epoch để vẽ đường validation accuracy theo epoch.
- **Lưu ý:** Quên model.eval() khiến dropout/batchnorm vẫn ở chế độ train làm kết quả val dao động sai. Nhớ gọi model.train() lại trước vòng huấn luyện tiếp theo. Lấy giá trị từ tensor 0 chiều bằng .item().
- *Nguồn: Week3Exercise1_MNIST_CNN.txt*


---

### 8.6 CNN — khối xây dựng

#### Lớp tích chập 2D (Conv2d)

- **Là gì:** Lớp `nn.Conv2d(in_channels, out_channels, kernel_size)` trượt một tập filter nhỏ (vd 3x3) trên ảnh, mỗi filter cho ra một feature map (kênh đầu ra).
- **Ý nghĩa & mục đích:** Là khối lõi của CNN. Thay vì nối mọi pixel như Linear, mỗi neuron chỉ nhìn một vùng nhỏ (local) nên bắt được đặc trưng cục bộ (cạnh, góc, đường) và dùng chung trọng số khắp ảnh. Dùng khi bài toán có cấu trúc không gian (ảnh) — vd percolation phụ thuộc tính cục bộ nên conv tốt hơn Dense.
- **Code:**

```python
nn.Conv2d(1, 6, 3)      # 1 kênh vào (xám) -> 6 kênh ra, filter 3x3
nn.Conv2d(3, 32, kernel_size=3, padding=1)  # ảnh màu 3 kênh vào
```
- **Visual:** Trực quan: mô tả filter 3x3x(depth) trượt qua ảnh, tại mỗi vị trí lấy tích vô hướng patch với filter rồi cộng bias -> một ô của feature map. Có thể xem gif convolution demo (input xanh, filter đỏ, output xanh lá) trong notebook visualize_activation. Xem kết quả bằng plt.imshow feature map.
- **Lưu ý:** Không có padding thì mỗi lớp 3x3 làm ảnh nhỏ đi 2 (28->26->24...). Phải theo dõi kích thước tensor để nối đúng với Linear phía sau. `in_channels` của lớp sau PHẢI bằng `out_channels` của lớp trước.
- *Nguồn: Week3Exercise1_MNIST_CNN.txt, Week3Exercise2_Percolation_CNN.txt, visualize_activation.txt*

#### Kênh vào/ra (in_channels / out_channels / filters)

- **Là gì:** `in_channels` = số kênh của tensor đầu vào (1 cho ảnh xám, 3 cho ảnh RGB); `out_channels` = số filter = số feature map đầu ra.
- **Ý nghĩa & mục đích:** Xác định độ sâu (depth) của tensor. Mỗi out_channel là một filter học một mẫu (template) riêng. Tăng out_channels = cho mạng học nhiều loại đặc trưng hơn; thường tăng dần theo độ sâu (32 -> 64 -> 128).
- **Code:**

```python
self.filters = 32
nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=3, padding=1)
nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=3, padding=1)
```
- **Visual:** Hình cột theo depth: nhiều neuron (vd 5) xếp theo chiều sâu cùng nhìn một vùng ảnh, mỗi neuron là một filter. Xem figures/depthcol.jpeg trong visualize_activation.
- **Lưu ý:** Lỗi phổ biến: đặt in_channels của lớp sau khác out_channels lớp trước -> lỗi shape khi forward.
- *Nguồn: visualize_activation.txt, Week3Exercise1_MNIST_CNN.txt*

#### Kích thước filter / kernel (kernel_size)

- **Là gì:** `kernel_size` là kích thước cửa sổ trượt của filter, vd 3 (3x3) hoặc 5 (5x5). Filter còn gọi là kernel.
- **Ý nghĩa & mục đích:** Quyết định vùng nhìn cục bộ của mỗi neuron. Với ảnh sâu D, một neuron 3x3 thực chất là 3x3xD trọng số (vd 3x3x3 = 27 trọng số cho ảnh RGB). Filter nhỏ (3x3) phổ biến vì rẻ và xếp chồng nhiều lớp vẫn phủ được vùng lớn.
- **Code:**

```python
self.filter_size = 3
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=self.filter_size, padding=1)
```
- **Visual:** Vẽ patch 3x3 trên ảnh và filter 3x3 tương ứng; tích vô hướng từng phần tử rồi cộng. Xem figures/cnn_dot_prod.png trong visualize_activation.
- **Lưu ý:** Filter càng lớn càng nhiều trọng số và chậm. Ảnh RGB thì filter tự động có chiều sâu = số kênh, không cần khai báo depth riêng.
- *Nguồn: visualize_activation.txt*

#### Padding (đệm viền)

- **Là gì:** `padding=1` thêm một viền số 0 quanh ảnh trước khi tích chập, giữ cho feature map ra cùng kích thước không gian với đầu vào (với filter 3x3).
- **Ý nghĩa & mục đích:** Dùng khi muốn giữ nguyên chiều cao/rộng qua các lớp conv, để dễ tính kích thước và không mất thông tin ở viền. Vd 8x8 -> vẫn 8x8 khi padding=1 với 3x3.
- **Code:**

```python
nn.Conv2d(1, 12, 3, padding=1)   # 8x8 -> 8x8
nn.Conv2d(12, 12, 3, padding=1)
```
- **Visual:** Vẽ ảnh gốc với một viền ô 0 bao quanh; filter trượt phủ được cả các ô rìa. Xem mô tả P=1 trong figures/3d_conv.gif của visualize_activation.
- **Lưu ý:** Không padding thì mỗi lớp 3x3 cắt mất 2 pixel mỗi chiều; quên điều này sẽ tính sai số đầu vào cho lớp Linear.
- *Nguồn: Week3Exercise2_Percolation_CNN.txt, visualize_activation.txt*

#### Stride (bước trượt)

- **Là gì:** `stride` là số pixel filter nhảy mỗi bước. stride=1 trượt từng ô; stride=2 nhảy 2 ô nên feature map ra nhỏ đi khoảng một nửa.
- **Ý nghĩa & mục đích:** Dùng để giảm kích thước không gian (downsample) ngay trong lớp conv, thay cho pooling. Công thức đầu ra mỗi chiều: (W - F + 2P)/S + 1.
- **Code:**

```python
nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
```
- **Visual:** Ví dụ trong visualize_activation: input W=5, F=3, S=2, P=1 -> output (5-3+2)/2+1 = 3. Vẽ filter nhảy 2 ô mỗi lần.
- **Lưu ý:** stride>1 làm mất chi tiết không gian; kết quả (W-F+2P)/S+1 phải ra số nguyên, nếu không PyTorch làm tròn xuống (floor) và kích thước sẽ khác dự tính.
- *Nguồn: visualize_activation.txt*

#### Hàm kích hoạt ReLU

- **Là gì:** ReLU giữ nguyên giá trị dương, ép giá trị âm về 0. Có hai cách dùng: lớp `nn.ReLU()` trong Sequential, hoặc hàm `F.relu(...)` trong forward.
- **Ý nghĩa & mục đích:** Thêm tính phi tuyến sau mỗi conv/linear; nếu không có thì nhiều lớp tuyến tính chồng lên nhau vẫn chỉ là một phép tuyến tính. Percolation cần hàm phi tuyến mạnh nên ReLU giữa các lớp là bắt buộc.
- **Code:**

```python
# cách 1: lớp trong Sequential
nn.Conv2d(1,12,3,padding=1), nn.ReLU()
# cách 2: hàm trong forward
x = F.relu(self.conv1(x))
```
- **Visual:** Vẽ đồ thị y=max(0,x): phẳng ở 0 khi x<0, đường thẳng khi x>0.
- **Lưu ý:** `nn.ReLU()` là class, PHẢI có ngoặc () để tạo instance khi đặt trong Sequential. `F.relu` thì gọi trực tiếp như hàm.
- *Nguồn: Week3Exercise2_Percolation_CNN.txt, visualize_activation.txt*

#### Max pooling (MaxPool2d)

- **Là gì:** `nn.MaxPool2d(2,2)` chia feature map thành các ô 2x2 và lấy giá trị lớn nhất mỗi ô, làm chiều cao/rộng giảm một nửa.
- **Ý nghĩa & mục đích:** Giảm kích thước không gian để bớt tính toán và cho mạng bất biến nhẹ với dịch chuyển. Đặt sau vài lớp conv. Lưu ý: trên percolation tác giả thấy max-pooling lại làm giảm nhẹ hiệu năng.
- **Code:**

```python
self.mp = nn.MaxPool2d(2,2)
x = self.mp(F.relu(self.bn2(self.conv2(x))))  # 32x32 -> 16x16
```
- **Visual:** Vẽ ô 2x2 -> chọn số lớn nhất -> một ô đầu ra. So sánh feature map trước/sau bằng plt.imshow.
- **Lưu ý:** Pooling làm mất thông tin vị trí chi tiết; không phải lúc nào cũng có lợi (percolation bị giảm accuracy). Nhớ cập nhật kích thước tensor cho lớp Linear sau đó.
- *Nguồn: visualize_activation.txt, Week3Exercise2_Percolation_CNN.txt*

#### Batch normalization 2D (BatchNorm2d)

- **Là gì:** `nn.BatchNorm2d(num_features)` chuẩn hoá đầu ra của lớp conv theo từng kênh (trừ trung bình, chia độ lệch chuẩn của batch) trước khi qua kích hoạt.
- **Ý nghĩa & mục đích:** Giúp huấn luyện mạng sâu ổn định và nhanh hơn, giảm phụ thuộc vào khởi tạo và learning rate. num_features phải bằng số kênh của conv ngay trước nó.
- **Code:**

```python
self.bn2 = nn.BatchNorm2d(self.filters*2)
x = self.mp(F.relu(self.bn2(self.conv2(x))))
```
- **Visual:** Vẽ histogram giá trị kích hoạt trước/sau BN: sau BN phân bố gọn quanh 0, std ~1.
- **Lưu ý:** Số kênh của BatchNorm2d phải khớp out_channels của conv trước. Khi suy luận PHẢI gọi model.eval() để BN dùng thống kê đã học (running mean/var) thay vì thống kê batch — nếu quên, dự đoán trên batch nhỏ sẽ chập chờn.
- *Nguồn: visualize_activation.txt*

#### Làm phẳng để nối Linear (Flatten)

- **Là gì:** `nn.Flatten()` (hoặc `x.reshape(-1, N)`) biến tensor nhiều chiều (kênh x cao x rộng) thành một vector phẳng để đưa vào lớp Linear.
- **Ý nghĩa & mục đích:** Conv làm việc với tensor không gian, nhưng lớp phân loại cuối (Linear) cần vector 1 chiều. Flatten là cầu nối giữa phần conv và phần dense.
- **Code:**

```python
nn.Flatten(),
nn.Linear(128, 12)   # 128 = số phần tử tensor sau conv
# hoặc trong forward:
x = x.reshape(-1, self.filters*4*8*8)
```
- **Visual:** Vẽ khối 3D (C x H x W) được duỗi thẳng thành một hàng dài.
- **Lưu ý:** Phải tính đúng số phần tử sau conv (kênh*cao*rộng) làm in_features cho Linear, nếu sai sẽ lỗi shape. Dùng mẹo chạy thử với torch.zeros để lấy số này. Ưu tiên `reshape(-1, N)` hơn `view(-1, N)` khi tensor có thể không liền mạch bộ nhớ.
- *Nguồn: Week3Exercise2_Percolation_CNN.txt, visualize_activation.txt*

#### Mẹo tính kích thước tensor bằng zeros

- **Là gì:** Chạy thử một phần mạng với tensor toàn số 0 đúng shape đầu vào, rồi xem `.shape` đầu ra để biết cần bao nhiêu phần tử cho lớp Linear.
- **Ý nghĩa & mục đích:** Khi dựng CNN phải theo dõi kích thước tensor qua từng lớp; tính tay dễ sai. Chạy thử là cách nhanh và chắc để lấy con số cho nn.Linear.
- **Code:**

```python
nn.Sequential(
    nn.Conv2d(1,12,3,padding=1), nn.ReLU(),
    nn.Conv2d(12,12,3,padding=1), nn.ReLU(),
    nn.Conv2d(12,2,3,padding=1), nn.ReLU()
)( torch.zeros(1,1,8,8) ).shape   # -> lấy tích các chiều làm in_features
```
- **Visual:** In ra shape trung gian, vd torch.Size([1, 2, 8, 8]) -> 2*8*8 = 128.
- **Lưu ý:** Nhớ đầu vào phải có đủ 4 chiều (batch, kênh, cao, rộng), vd torch.zeros(1,1,8,8).
- *Nguồn: Week3Exercise2_Percolation_CNN.txt*

#### Định nghĩa CNN bằng subclass nn.Module

- **Là gì:** Tạo lớp con của `nn.Module`, khai báo các lớp trong `__init__` và viết luồng dữ liệu trong `forward`. Có thể gom lớp bằng nn.Sequential hoặc để rời và gọi trong forward.
- **Ý nghĩa & mục đích:** Cách chuẩn để đóng gói kiến trúc: mọi thứ (khởi tạo trọng số + forward) ở một chỗ, và mỗi lần gọi class() là một mạng khởi tạo mới sạch. Tránh lỗi quên reset trọng số khi thử nghiệm nhiều lần.
- **Code:**

```python
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
        self.mp = nn.MaxPool2d(2, 2)          # 32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(128*8*8, 10)
    def forward(self, x):
        x = self.mp(F.relu(self.conv1(x)))    # -> 16x16
        x = self.mp(F.relu(self.conv2(x)))    # -> 8x8
        x = x.reshape(-1, 128*8*8)
        return self.fc1(x)
```
- **Visual:** Sơ đồ kiến trúc: ảnh -> conv -> relu -> pool -> ... -> flatten -> dense -> lớp ra. Xem figures/cnn_arch.png trong visualize_activation.
- **Lưu ý:** Phải gọi super().__init__() trước (source dùng dạng cũ super(CNNModel, self).__init__(); dạng super().__init__() gọn hơn, tương đương). Lỗi hay gặp: train lại nhiều lần trên cùng một instance mà quên tạo mới -> trọng số không reset. Khi LƯU model, nên dùng torch.save(model.state_dict(), path) rồi load_state_dict(...) chứ ĐỪNG torch.save cả object model (cách này giòn, phụ thuộc đường dẫn class khi load lại).
- *Nguồn: visualize_activation.txt, Week3Exercise1_MNIST_CNN.txt, Week3Exercise2_Percolation_CNN.txt*

#### Xâu chuỗi lớp bằng nn.Sequential

- **Là gì:** `nn.Sequential(layer1, layer2, ...)` nối nhiều lớp thành một khối, dữ liệu chạy qua lần lượt theo thứ tự khai báo.
- **Ý nghĩa & mục đích:** Cách nhanh gọn để dựng mạng thẳng (không rẽ nhánh). Gói cả conv, relu, pool, flatten, linear vào một self.layers rồi forward chỉ cần gọi một dòng.
- **Code:**

```python
self.layers = nn.Sequential(
    nn.Conv2d(1,12,3,padding=1), nn.ReLU(),
    nn.Conv2d(12,2,3,padding=1), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(128,12), nn.ReLU(),
    nn.Linear(12,1), nn.Sigmoid())
# forward: x = self.layers(x)
```
- **Visual:** Vẽ chuỗi hộp nối tiếp nhau bằng mũi tên, dữ liệu chảy một chiều.
- **Lưu ý:** Chỉ hợp cho mạng tuyến tính thẳng; nếu cần rẽ nhánh (skip connection) hoặc lấy đầu ra giữa chừng thì phải để lớp rời và viết forward thủ công.
- *Nguồn: Week3Exercise2_Percolation_CNN.txt, Week3Exercise1_MNIST_CNN.txt*

#### Filter/kernel như bộ dò mẫu (feature detector)

- **Là gì:** Mỗi filter trong lớp conv học một template; ảnh chứa mẫu giống filter sẽ làm đầu ra (activation) của filter đó lớn. Lớp nông học mẫu đơn giản (đường, cạnh), lớp sâu học mẫu phức tạp hơn.
- **Ý nghĩa & mục đích:** Cách hiểu trực giác vì sao CNN hoạt động: mạng tự học các bộ dò đặc trưng theo tầng, từ đơn giản đến phức tạp, thay vì phải thiết kế tay.
- **Code:**

```python
# xem số filter của một lớp
num_filters = model.conv4.out_channels
print('Layer conv4 has', num_filters, 'filters')
```
- **Visual:** Trực quan bằng activation maximization: tạo ảnh ngẫu nhiên rồi gradient ascent để tối đa đầu ra một filter -> ảnh cho thấy filter thích mẫu gì. Hoặc xem top feature map kích hoạt mạnh nhất cho một ảnh input.
- **Lưu ý:** Một số filter bị 'kẹt ở 0' (loss <= eps) -> ảnh vô nghĩa, nên bỏ qua. Ảnh CIFAR-10 độ phân giải thấp nên mẫu học được không đẹp/rõ. Lưu ý code activation-maximization trong source dùng torch.autograd.Variable ĐÃ BỎ — PyTorch hiện đại tạo tensor cần gradient bằng torch.tensor(..., requires_grad=True) trực tiếp, không cần Variable.
- *Nguồn: visualize_activation.txt*

#### Lấy đầu ra lớp giữa bằng forward hook

- **Là gì:** `layer.register_forward_hook(fn)` gắn một hàm được gọi tự động mỗi lần lớp đó chạy forward, cho phép lưu lại (output) của lớp mà không sửa forward gốc.
- **Ý nghĩa & mục đích:** Dùng để soi hoạt động bên trong CNN: lấy feature map của từng lớp conv để trực quan hoá xem mỗi lớp 'nhìn' ảnh thế nào.
- **Code:**

```python
activation = {}
def get_activation(name):
    def hook(module, inp, output):
        activation[name] = F.relu(output)
    return hook

model.conv1.register_forward_hook(get_activation('conv1'))
model.conv4.register_forward_hook(get_activation('conv4'))
# sau khi model(input) chạy, đọc activation['conv1']
```
- **Visual:** Sau forward, plt.imshow từng activation[layer][filter] để xem feature map; sắp xếp theo độ kích hoạt lấy top 10.
- **Lưu ý:** Hook chỉ điền dữ liệu SAU khi chạy một forward pass; đọc activation trước khi gọi model(input) sẽ rỗng. Nhớ gỡ hook (handle.remove()) khi không dùng nữa để tránh giữ tham chiếu/tốn bộ nhớ.
- *Nguồn: visualize_activation.txt*

#### Liệt kê các lớp trong model (named_modules / named_children)

- **Là gì:** `model.named_modules()` duyệt qua tên và đối tượng của từng lớp con trong mạng (named_children chỉ duyệt các lớp con trực tiếp một cấp).
- **Ý nghĩa & mục đích:** Dùng để biết tên chính xác của các lớp (conv1, fc2...) nhằm đăng ký hook, truy cập thuộc tính (out_channels), hoặc kiểm tra kiến trúc.
- **Code:**

```python
for name, layer in model.named_modules():
    print(name, layer)
```
- **Visual:** In ra danh sách tên lớp kèm mô tả (Conv2d(...), Linear(...)) — bản đồ kiến trúc dạng chữ.
- **Lưu ý:** Tên lớp phụ thuộc cách đặt thuộc tính trong __init__; nếu dùng Sequential thì tên là chỉ số (layers.0, layers.1...).
- *Nguồn: visualize_activation.txt*

#### Ý tưởng convolution 1x1

- **Là gì:** Convolution với kernel 1x1: trộn thông tin giữa các kênh tại mỗi vị trí mà không nhìn vùng lân cận không gian.
- **Ý nghĩa & mục đích:** Được nêu như một hướng thử để cải thiện mô hình percolation (giảm/đổi số kênh, thêm phi tuyến rẻ). Là một khối xây dựng nhẹ đáng cân nhắc.
- **Code:**

```python
nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 1x1 conv
```
- **Visual:** Vẽ filter 1x1 trượt qua từng pixel, chỉ kết hợp theo chiều sâu (kênh).
- **Lưu ý:** Trong khoá học đây mới là gợi ý ('Perhaps some 1x1 convolutions might improve performance? I haven't tried them'), chưa có kết quả kiểm chứng.
- *Nguồn: Week3Exercise2_Percolation_CNN.txt*

#### Kết nối cục bộ và trường tiếp nhận (local connectivity / receptive field)

- **Là gì:** Mỗi neuron conv chỉ nối với một vùng nhỏ của đầu vào theo không gian (vd 3x3) nhưng nối đủ theo chiều sâu (tất cả kênh màu). Vùng đó là trường tiếp nhận của neuron.
- **Ý nghĩa & mục đích:** Giải thích vì sao CNN ít trọng số hơn Dense và bắt được đặc trưng cục bộ. output[i,j,:] chỉ phụ thuộc patch input quanh (i,j). Xếp nhiều lớp thì trường tiếp nhận lớn dần, phủ được đặc trưng rộng hơn.
- **Code:**

```python
# neuron 3x3 trên ảnh RGB = 3x3x3 = 27 trọng số + 1 bias
# vector output[i, j, :] đến từ patch input[i-1:i+1, j-1:j+1, :]
```
- **Visual:** Vẽ một neuron nối tới một ô 3x3 trên ảnh (không nối toàn ảnh). Xem figures/neuron_model.jpeg và cnn_mapping.jpg trong visualize_activation.
- **Lưu ý:** Kết nối cục bộ nghĩa là conv một mình không thấy được quan hệ giữa các pixel ở xa nhau trong một lớp — cần chồng nhiều lớp (hoặc pooling) để mở rộng trường tiếp nhận; đây là lý do percolation (phụ thuộc đường dài) khó.
- *Nguồn: visualize_activation.txt, Week3Exercise2_Percolation_CNN.txt*

#### Chế độ train/eval của model (model.train / model.eval)

- **Là gì:** `model.train()` và `model.eval()` chuyển toàn mạng giữa hai chế độ. Chúng KHÔNG tự tính gradient hay cập nhật trọng số — chỉ đổi cách hành xử của các lớp phụ thuộc chế độ như Dropout và BatchNorm.
- **Ý nghĩa & mục đích:** Bắt buộc dùng khi mạng có BatchNorm hoặc Dropout: lúc huấn luyện cần train() để BN dùng thống kê batch và Dropout bật; lúc validate/predict cần eval() để BN dùng thống kê đã học và Dropout tắt. Quên chuyển eval() là lỗi âm thầm làm kết quả suy luận sai lệch.
- **Code:**

```python
model.train()   # trước vòng lặp huấn luyện
# ... forward, loss.backward(), optimizer.step() ...

model.eval()    # trước khi đánh giá / dự đoán
with torch.no_grad():
    outputs = model(images)
```
- **Visual:** Sơ đồ: cùng một mạng, công tắc train/eval đổi hành vi của khối BN và Dropout; các lớp conv/linear không đổi.
- **Lưu ý:** model.eval() và with torch.no_grad() là HAI việc khác nhau: eval() đổi hành vi BN/Dropout, no_grad() tắt tính gradient (tiết kiệm bộ nhớ). Khi suy luận thường cần cả hai. Sau khi load_state_dict để dự đoán, nhớ gọi eval().
- *Nguồn: Week3Exercise1_MNIST_CNN.txt, visualize_activation.txt*

#### Dropout (nn.Dropout / nn.Dropout2d)

- **Là gì:** Dropout ngẫu nhiên tắt (đưa về 0) một phần neuron trong lúc huấn luyện, theo xác suất p. `nn.Dropout(p)` cho vector (sau Linear); `nn.Dropout2d(p)` tắt cả feature map (kênh) sau conv.
- **Ý nghĩa & mục đích:** Là một cách regularization để chống overfit: buộc mạng không phụ thuộc vào một vài neuron riêng lẻ. Notebook percolation gợi ý thử dropout (cùng l1/l2) khi mô hình khớp train tốt nhưng lỗi validation cao.
- **Code:**

```python
nn.Dropout(0.5)        # sau lớp Linear
nn.Dropout2d(0.25)     # sau lớp Conv (tắt cả kênh)
# trong Sequential:
# ..., nn.ReLU(), nn.Dropout(0.5), nn.Linear(100, 1)
```
- **Visual:** Vẽ một lớp neuron, một số neuron bị gạch chéo (tắt) ở mỗi bước huấn luyện, thay đổi ngẫu nhiên qua các batch.
- **Lưu ý:** Dropout CHỈ hoạt động ở chế độ train(); ở eval() nó tự tắt và không cần chỉnh tay tỉ lệ. Vì vậy phải gọi model.eval() khi đánh giá, nếu không kết quả sẽ nhiễu. p quá lớn (vd 0.7) có thể làm mạng khó khớp cả train.
- *Nguồn: Week3Exercise2_Percolation_CNN.txt, Week3Exercise1_MNIST_CNN.txt*


---

### 8.7 RNN / Mô hình chuỗi

#### Lớp LSTM (nn.LSTM)

- **Là gì:** Lớp mạng hồi tiếp Long Short-Term Memory trong PyTorch, xử lý một chuỗi vector và trả về (output tất cả bước thời gian, hidden state cuối). Có cổng nhớ (cell state) nên giữ được thông tin xa.
- **Ý nghĩa & mục đích:** Dùng khi dữ liệu là chuỗi (chuỗi ký tự, câu văn) và cần nhớ trạng thái qua nhiều bước. LSTM giảm nhẹ vanishing gradient của RNN thường, nên học được phụ thuộc xa (nhớ ký tự sau R để so với ký tự sau S, hoặc khớp ngoặc lồng sâu).
- **Code:**

```python
self.lstm = nn.LSTM(embedding_dim, hidden_dim, 1, batch_first=True)
# forward:
lstm_out, hidden = self.lstm(embeds, hidden)
# lstm_out: (batch, seq_len, hidden_dim); hidden = (h_n, c_n)
```
- **Visual:** Vẽ heatmap output LSTM theo từng bước chuỗi: plt.imshow(lstm_outputs[0,:,:].detach().cpu().numpy().transpose()); plt.colorbar(). Trục x = vị trí ký tự, trục y = các neuron hidden; nhìn được neuron nào 'bật' khi gặp ký tự quan trọng.
- **Lưu ý:** batch_first=True thì tensor là (batch, seq, feat); nếu quên, thứ tự chiều bị sai. hidden là TUPLE (h_n, c_n) chứ không phải 1 tensor. embedding_dim đầu vào phải khớp số chiều embedding. Nếu không truyền hidden, LSTM tự khởi tạo bằng 0.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt; 3 - IMDB_sentiment_analysis.txt*

#### Lớp RNN đơn giản (nn.RNN)

- **Là gì:** Lớp hồi tiếp cơ bản (Elman RNN): mỗi bước lấy input + hidden trước đó, qua một phép biến đổi tuyến tính + tanh. Không có cổng nhớ như LSTM.
- **Ý nghĩa & mục đích:** Dùng làm baseline để so với LSTM. Trong khoá, RNN học kém hơn hẳn LSTM trên bài phân loại cảm xúc IMDB vì khó nhớ phụ thuộc xa (gradient tiêu biến trên chuỗi dài 500 từ).
- **Code:**

```python
if model_type == 'rnn':
    self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
else:
    self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
```
- **Visual:** Vẽ 2 đường val_accuracy của RNN và LSTM trên cùng biểu đồ để thấy LSTM cao hơn rõ. Dùng plot_history(history_rnn) rồi plot_history(history_lstm).
- **Lưu ý:** RNN.forward trả về (output, h_n) chỉ 1 hidden tensor; còn LSTM trả (output, (h_n, c_n)) tuple. Ở IMDB cả hai đều gọi self.rnn(embeds) rồi bỏ hidden bằng '_' nên chạy chung được; nhưng nếu cần dùng hidden phải cẩn thận khác biệt này. RNN dễ bị vanishing gradient trên chuỗi dài.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Lớp Embedding (nn.Embedding)

- **Là gì:** Bảng tra cứu biến mỗi số nguyên (id token) thành một vector đặc trưng học được. Đầu vào là tensor số nguyên, đầu ra thêm một chiều embedding_dim.
- **Ý nghĩa & mục đích:** Dùng để biến chuỗi id từ/ký tự thành vector cho RNN xử lý. Với văn bản (IMDB), embedding được HỌC trong lúc train nên tự tìm biểu diễn từ có ý nghĩa.
- **Code:**

```python
self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
# forward:
embeds = self.word_embeddings(sentence)  # (batch, seq) -> (batch, seq, embedding_dim)
```
- **Visual:** Có thể xem shape: model.word_embeddings.weight.shape = (vocab_size, embedding_dim). Trực quan hoá bằng cách chiếu embedding các từ về 2D (PCA/t-SNE) — nhưng khoá không làm, chỉ in shape.
- **Lưu ý:** vocab_size phải bao trùm mọi id (khoá dùng len(vocab)+1 để chừa id 0 làm padding). Đầu vào PHẢI là kiểu Long/int64, nếu là float sẽ lỗi. Có padding_idx nếu muốn cố định vector padding = 0 (khoá không dùng).
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Embedding one-hot cố định (Embedding.from_pretrained + np.eye)

- **Là gì:** Tạo ma trận embedding là ma trận đơn vị (one-hot) rồi nạp vào Embedding và đóng băng (freeze), nên mỗi ký tự thành một vector one-hot không đổi.
- **Ý nghĩa & mục đích:** Dùng khi vocab nhỏ (vài ký tự như a,b,R,S,ngoặc) và không muốn học embedding — chỉ cần mã hoá phân biệt các ký tự. Giữ mô hình đơn giản, tập trung xem LSTM học logic chuỗi.
- **Code:**

```python
embedding_matrix = torch.from_numpy(np.eye(n_chars)).float()
embedding_matrix.requires_grad = False
char_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
```
- **Visual:** In embedding_matrix ra: là ma trận đơn vị n_chars x n_chars. char_embedding(some_int_tensor) cho thấy mỗi số thành một hàng one-hot.
- **Lưu ý:** freeze=True để không train embedding. Kích thước n_chars = len(c2i)+1 (chừa 0 cho padding); tính thiếu 1 sẽ lỗi index. np.eye trả float64 nên phải .float() (float32) trước khi nạp, không thì lệch kiểu với phần còn lại của mạng.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt*

#### Hidden state ban đầu của LSTM (init_hidden)

- **Là gì:** Hàm tạo trạng thái ẩn ban đầu (h0, c0) toàn số 0 với đúng shape (n_layers, batch_size, hidden_dim) cho LSTM.
- **Ý nghĩa & mục đích:** LSTM cần trạng thái khởi đầu. Tạo bằng 0 ở đầu mỗi epoch (hoặc mỗi chuỗi mới) để mô hình không mang thông tin cũ sang.
- **Code:**

```python
def init_hidden(self, batch_size):
    weight = next(self.parameters()).data
    hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
              weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
    return hidden
```
- **Visual:** In h ra thấy tuple 2 tensor toàn 0 shape (1, batch_size, hidden_dim). Không có hình đặc thù.
- **Lưu ý:** Phải là TUPLE 2 tensor (h cho hidden, c cho cell) vì là LSTM; RNN chỉ cần 1 tensor. batch_size của batch cuối có thể nhỏ hơn — DataLoader không drop_last, nên init cứng theo batch_size chuẩn sẽ lệch chiều (dùng x.size(0) trong forward). Idiom weight.new(...).zero_() là cách CŨ; hiện đại nên dùng torch.zeros(n_layers, batch, hidden, device=device).
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt*

#### Ngắt đồ thị hidden state (detach hidden / truncated BPTT)

- **Là gì:** Trước mỗi batch, thay hidden bằng bản sao chỉ giữ .data để cắt liên kết với đồ thị tính toán của batch trước.
- **Ý nghĩa & mục đích:** Ngăn backprop lan ngược vô hạn qua các batch (truncated backpropagation through time). Tránh lỗi 'backward qua đồ thị đã giải phóng' và tránh tốn bộ nhớ.
- **Code:**

```python
h = tuple(e.data for e in h)
# đặt ngay trước model.zero_grad() / model(inputs, h) trong vòng lặp batch
# cách hiện đại:
h = tuple(e.detach() for e in h)
```
- **Visual:** Không có hình. Khái niệm về luồng gradient; có thể minh hoạ bằng sơ đồ chuỗi batch với 'kéo' gradient bị cắt giữa các batch.
- **Lưu ý:** Quên detach -> RuntimeError 'Trying to backward through the graph a second time' (đồ thị đã bị free). Dùng .data né autograd nhưng là idiom CŨ và không an toàn; hiện đại nên dùng .detach(). Với RNN ở IMDB không giữ hidden giữa batch nên không cần bước này.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt*

#### Lấy output bước cuối để phân loại chuỗi (out[:,-1])

- **Là gì:** Kiểu forward: gộp output mọi bước qua lớp Linear + Sigmoid, reshape về (batch, -1) rồi lấy cột cuối cùng làm dự đoán cho cả chuỗi.
- **Ý nghĩa & mục đích:** Bài phân loại chuỗi (câu tích cực/tiêu cực, chuỗi hợp lệ/không) cần MỘT nhãn cho cả chuỗi. Bước thời gian cuối của LSTM đã 'đọc' toàn bộ chuỗi nên dùng nó để quyết định.
- **Code:**

```python
lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
out = self.fc(lstm_out)
out = self.sigmoid(out)
out = out.view(batch_size, -1)
out = out[:,-1]   # lấy output ở bước cuối
```
- **Visual:** Không có hình riêng. Có thể minh hoạ bằng chuỗi ô, ô cuối được tô đậm = output dùng để phân loại.
- **Lưu ý:** contiguous() cần trước view() vì sau LSTM tensor có thể không liền mạch bộ nhớ. Comment '# selects the last' trong source gây hiểu lầm: chính out[:,-1] mới chọn bước cuối, không phải view. Cách này giả định các chuỗi cùng độ dài (đã pad) và bước cuối = thông tin đầy đủ. Cách sạch hơn: lấy trực tiếp lstm_out[:, -1, :] rồi mới qua Linear.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 3 - IMDB_sentiment_analysis.txt*

#### Mô hình phân loại chuỗi (Embedding + RNN/LSTM + Linear)

- **Là gì:** Lớp nn.Module ghép 3 phần: Embedding (mã hoá token) -> RNN hoặc LSTM (đọc chuỗi) -> Linear + Sigmoid (ra xác suất). Chọn lõi hồi tiếp bằng tham số model_type. Kiến trúc chuẩn cho phân loại chuỗi nhị phân.
- **Ý nghĩa & mục đích:** Khuôn tái sử dụng cho mọi bài 'chuỗi -> 1 nhãn': nhớ ký tự, khớp ngoặc, cảm xúc phim. Đổi Embedding / model_type / kích thước là dùng lại được.
- **Code:**

```python
class SentimentAnalyser(nn.Module):
    def __init__(self, model_type, embedding_dim, hidden_dim, vocab_size, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if model_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, sentence):
        bs = sentence.shape[0]
        embeds = self.word_embeddings(sentence)
        rnn_out, _ = self.rnn(embeds)
        rnn_out = rnn_out.contiguous().view(-1, self.hidden_dim)
        out = self.sigmoid(self.hidden(rnn_out))
        return out.view(bs, -1)[:,-1]
```
- **Visual:** Vẽ sơ đồ khối: token id -> Embedding -> RNN/LSTM (cuộn theo thời gian) -> Linear -> Sigmoid -> xác suất. Có thể vẽ bằng SVG/box diagram.
- **Lưu ý:** num_classes ở khoá = len(unique(y)) = 2, nhưng vì view(bs,-1)[:,-1] nên đầu ra thu về 1 số dùng với BCELoss (cách này hơi lắt léo). Truyền model_type='rnn' hay khác ('lstm') để chọn lõi hồi tiếp. Chú ý: lớp Linear đặt tên self.hidden dễ nhầm với hidden state của RNN.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt; 1 - What_can_recurrent_nets_learn.txt*

#### Padding chuỗi ở đầu bằng số 0 (string_to_int_vec)

- **Là gì:** Chuyển chuỗi thành vector số nguyên độ dài cố định pad_length, chèn các số 0 ở ĐẦU nếu chuỗi ngắn hơn (pre-padding).
- **Ý nghĩa & mục đích:** Mạng cần input cùng độ dài để gộp thành batch (tensor chữ nhật). Pad ở đầu để phần nội dung thật nằm sát bước cuối — nơi ta lấy output phân loại.
- **Code:**

```python
def string_to_int_vec(s, pad_length, code_dict):
    slen = len(s)
    assert slen <= pad_length
    v = np.zeros([pad_length])
    startx = pad_length - slen   # pad ở đầu, không phải cuối
    for i in range(slen):
        v[startx + i] = code_dict[list(s)[i]]
    return v
```
- **Visual:** In một hàng của mảng đã mã hoá: các số 0 ở đầu rồi đến id ký tự. Ví dụ 'abRSab' pad_length=10 -> [0,0,0,0,1,2,...].
- **Lưu ý:** 0 được dành riêng làm ký hiệu padding nên bảng mã ký tự phải đánh số từ 1. assert slen<=pad_length: chuỗi dài hơn pad_length sẽ lỗi (cần cắt trước). Pre-padding (ở đây) khác post-padding của pad_sequence trong IMDB; vị trí padding ảnh hưởng ý nghĩa của out[:,-1].
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt*

#### Đệm chuỗi bằng pad_sequence (torch.nn.utils.rnn)

- **Là gì:** Hàm PyTorch gộp danh sách tensor độ dài khác nhau thành một tensor 2D bằng cách đệm giá trị padding cho bằng độ dài chuỗi dài nhất.
- **Ý nghĩa & mục đích:** Cách tiện để pad hàng loạt câu (đã token hoá) về cùng độ dài trong bài IMDB, thay vì tự viết vòng lặp.
- **Code:**

```python
from torch.nn.utils.rnn import pad_sequence
sequences = [torch.tensor(vocab(tokenizer(text)[0:maxlen])) for text in texts]
data = pad_sequence(sequences, True, padding_value=0)  # đối số thứ 2 = batch_first
```
- **Visual:** In data.shape = (số câu, maxlen). Có thể vẽ histogram độ dài câu gốc để thấy vì sao cần cắt maxlen=500.
- **Lưu ý:** pad_sequence pad ở CUỐI (mặc định), khác với string_to_int_vec pad ở đầu. Đối số thứ 2 True = batch_first. Nhớ cắt [0:maxlen] TRƯỚC khi tạo tensor để câu quá dài không làm tensor phình to. Muốn RNN bỏ qua phần pad đúng cách thì kết hợp pack_padded_sequence (khoá không dùng).
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Tokenizer tiếng Anh cơ bản (torchtext get_tokenizer)

- **Là gì:** Hàm tách một câu thành danh sách token (từ, dấu câu) theo quy tắc 'basic_english' (chữ thường + tách dấu).
- **Ý nghĩa & mục đích:** Bước đầu tiên xử lý văn bản: biến câu thô thành list từ để rồi ánh xạ sang id. Chuẩn hoá về chữ thường giúp gom từ.
- **Code:**

```python
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer("You can now install TorchText using pip!")
# ['you','can','now','install','torchtext','using','pip','!']
```
- **Visual:** In list token của một câu để thấy cách tách. Không có hình đồ hoạ.
- **Lưu ý:** DEPRECATION: torchtext đã NGỪNG phát triển và bị loại khỏi các bản PyTorch mới (bản cũ cảnh báo, bản rất mới không import được). Phải !pip install torchtext bản khớp torch, dễ lỗi version. Thay thế hiện đại: tokenizer của spaCy hoặc HuggingFace tokenizers.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Xây từ điển từ vựng (build_vocab_from_iterator)

- **Là gì:** Duyệt qua toàn bộ token của tập dữ liệu để lập từ điển ánh xạ từ -> id, có lọc theo tần suất tối thiểu và thêm token đặc biệt.
- **Ý nghĩa & mục đích:** Tạo bộ từ vựng cố định cho mô hình. Bỏ từ hiếm (min_freq) để giảm kích thước vocab và nhiễu; thêm <unk> để xử lý từ lạ khi gặp lúc test.
- **Code:**

```python
from torchtext.vocab import build_vocab_from_iterator
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)
vocab = build_vocab_from_iterator(yield_tokens(iter(texts)), min_freq=30, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
# dùng: vocab(tokenizer(text)) -> list id
```
- **Visual:** In len(vocab) để biết số từ giữ lại. Có thể vẽ histogram tần suất từ để chọn min_freq.
- **Lưu ý:** Phải gọi set_default_index(vocab['<unk>']) nếu không sẽ lỗi RuntimeError khi gặp từ ngoài từ điển. min_freq cao -> vocab nhỏ, mất thông tin; thấp -> vocab to, chậm. Xây vocab CHỈ từ tập train để tránh rò rỉ. DEPRECATION: torchtext.vocab nằm trong gói torchtext đã ngừng bảo trì — code cũ có thể không chạy trên torch mới.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Bảng mã ký tự <-> số (char_to_integers)

- **Là gì:** Hàm tạo 2 dict: ký tự -> số nguyên (đánh từ 1) và số -> ký tự, dựa trên tập ký tự xuất hiện trong chuỗi.
- **Ý nghĩa & mục đích:** Mã hoá cấp ký tự cho các bài chuỗi tổng hợp (a,b,R,S,ngoặc). Có cả chiều ngược để giải mã kiểm tra dữ liệu.
- **Code:**

```python
def char_to_integers(mystring):
    charlist = list(set(list(mystring)))
    nums = range(1, len(charlist)+1)
    c2ndict, n2cdict = dict(), dict()
    for c, n in zip(charlist, nums):
        c2ndict[c] = n; n2cdict[n] = c
    return c2ndict, n2cdict
# c2i, i2c = char_to_integers(''.join(training_strings))
```
- **Visual:** In c2i thấy {'a':1,'b':2,'R':3,...}. Kiểm tra bằng int_vec_to_string để giải mã ngược lại đúng chuỗi gốc.
- **Lưu ý:** Đánh số từ 1 vì 0 để dành cho padding; n_chars = len(c2i)+1. set() KHÔNG có thứ tự nên id có thể đổi giữa các lần chạy — phải build 1 lần rồi dùng lại cùng dict cho cả train/test (khoá build lại từ toàn bộ training_strings để chắc đủ ký tự).
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt*

#### Giải mã ngược để kiểm tra dữ liệu (int_vec_to_string)

- **Là gì:** Hàm biến vector số nguyên trở lại chuỗi ký tự, bỏ qua các số 0 (padding).
- **Ý nghĩa & mục đích:** Công cụ kiểm tra: sau khi mã hoá + pad, giải mã ngược phải ra đúng chuỗi gốc. Bắt lỗi mã hoá sai sớm.
- **Code:**

```python
def int_vec_to_string(v, i2c):
    charlist = []
    for i in range(v.shape[0]):
        if v[i] > 0:
            charlist.append(i2c[v[i]])
    return ''.join(charlist)
# check: ss[2], int_vec_to_string(data[2,:], i2c), ss_labels[2]
```
- **Visual:** So sánh cạnh nhau chuỗi gốc và chuỗi giải mã: phải trùng nhau. Không có hình.
- **Lưu ý:** Bỏ qua v[i]==0 vì đó là padding; nếu ký tự thật bị đánh id 0 sẽ mất khi giải mã. i2c phải là ĐÚNG dict đã dùng lúc mã hoá. Key i2c là số float (v[i] từ mảng float) nên dict phải nhận đúng kiểu key — ở đây chạy được vì float khớp, nhưng nên ép int(v[i]) cho chắc.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt*

#### Sinh dữ liệu chuỗi kiểu 'nhớ trạng thái' (generate_RS_memory_strings)

- **Là gì:** Hàm sinh chuỗi ngẫu nhiên có chèn 'R' và 'S'; nhãn 1 nếu ký tự sau S trùng ký tự sau R (nhớ đúng), nhãn 0 nếu khác.
- **Ý nghĩa & mục đích:** Tạo bài kiểm tra khả năng NHỚ của LSTM: mạng phải nhớ ký tự sau R rồi so với ký tự sau S cách đó vài bước. Thăm dò xem RNN học được phụ thuộc xa tới đâu.
- **Code:**

```python
def generate_RS_memory_strings(n_strings, string_length, charset):
    char_list = list(charset)
    # đặt R ở Rpos1, S ở Rpos2 (cách nhau >= 2)
    if label:
        this_string_list[Rpos2+1] = this_string_list[Rpos1+1]   # nhớ đúng
    else:
        while this_string_list[Rpos2+1] == this_string_list[Rpos1+1]:
            this_string_list[Rpos2+1] = random.choice(char_list)
    return string_list, label_list
```
- **Visual:** In vài chuỗi + nhãn để thấy quy luật: nhãn 1 = ký tự sau R và sau S giống nhau. Vẽ heatmap output LSTM trên một chuỗi để xem neuron 'nhớ'.
- **Lưu ý:** R và S phải cách nhau >=2 để có ký tự ngay sau. Chuỗi ngắn (length 10) với charset nhỏ (chỉ 'ab') khiến train/test TRÙNG rất nhiều — cần kiểm tra overlap (khoá thừa nhận 'quite an overlap'). Đây là dữ liệu tổng hợp để thăm dò, không phải bài thực tế.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt*

#### Sinh dữ liệu ngoặc khớp/sai (random_matching_brackets + make_good_and_bad_brackets)

- **Là gì:** Bộ hàm đệ quy sinh chuỗi ngoặc lồng nhau khớp đúng, rồi cố ý lật MỘT ngoặc để tạo chuỗi sai; nhãn 1 = khớp, 0 = sai.
- **Ý nghĩa & mục đích:** Bài kiểm tra khả năng học ngữ pháp lồng nhau (đếm/khớp ngoặc). Học nhanh hơn bài nhớ ký tự và tổng quát hoá tốt sang chuỗi dài, lồng sâu hơn tập train.
- **Code:**

```python
def random_matching_brackets(p, q, s=[]):
    while np.random.rand() < p:
        s.append('(')
        if np.random.rand() < q:
            s = random_matching_brackets(p, q, s=s)
        s.append(')')
    return s

def make_good_brackets_bad(s):
    x = random.randint(0, len(s)-1)
    s[x] = ')' if s[x] == '(' else '('
    return s
```
- **Visual:** In vài chuỗi ngoặc + nhãn. Vẽ heatmap output LSTM trên chuỗi ngoặc lồng để xem neuron nào theo dõi độ sâu lồng.
- **Lưu ý:** Tham số mặc định s=[] là danh sách MUTABLE — dễ dính bug 'default argument dùng chung' giữa các lần gọi; source né bằng cách LUÔN truyền s=[] tường minh. make_good_and_bad_brackets lọc len trong [5,20]; p, q điều khiển độ dài/độ lồng.
- *Nguồn: 2 - What_can_RNN_learn_with_brackets.txt*

#### Kiểm tra chồng lấn train/test (set intersection)

- **Là gì:** Dùng phép giao tập hợp để đếm số chuỗi xuất hiện ở cả tập train lẫn test.
- **Ý nghĩa & mục đích:** Với dữ liệu tổng hợp, số chuỗi khả dĩ hữu hạn nên train và test dễ trùng nhiều -> accuracy cao GIẢ vì mô hình chỉ 'thuộc lòng'. Đây là kiểm tra rò rỉ dữ liệu (data leakage).
- **Code:**

```python
train_string_set = set(training_strings)
test_string_set = set(test_strings)
overlap = train_string_set.intersection(test_string_set)
len(train_string_set), len(overlap), len(test_string_set)
```
- **Visual:** Vẽ Venn diagram 2 tập với phần giao. Hoặc chỉ in 3 con số để thấy overlap lớn cỡ nào.
- **Lưu ý:** Overlap lớn = val accuracy không phản ánh khả năng tổng quát hoá thật (source thừa nhận 'quite an overlap' ở bài nhớ). Cách khắc phục: tăng độ dài chuỗi / charset để không gian lớn hơn, hoặc loại chuỗi test trùng train.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt*

#### Đóng gói dữ liệu (TensorDataset + DataLoader)

- **Là gì:** TensorDataset gộp tensor input và nhãn thành cặp (x,y); DataLoader chia thành batch, có thể xáo trộn (shuffle).
- **Ý nghĩa & mục đích:** Chuẩn PyTorch để duyệt dữ liệu theo batch trong vòng lặp train. shuffle=True cho train giúp mỗi epoch thứ tự khác nhau, học ổn định hơn.
- **Code:**

```python
train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
test_data  = TensorDataset(torch.LongTensor(x_val), torch.LongTensor(y_val))
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader  = DataLoader(test_data, shuffle=True, batch_size=batch_size)
```
- **Visual:** In train_data[7] thấy cặp (tensor chuỗi, nhãn). Không có hình.
- **Lưu ý:** Phải dùng LongTensor cho chuỗi id vì Embedding cần int64. Batch cuối có thể nhỏ hơn batch_size -> lỗi nếu init_hidden cứng theo batch_size chuẩn (dùng x.size(0) trong forward để lấy đúng, hoặc drop_last=True). Test không nhất thiết cần shuffle (source ghi chú vậy).
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 3 - IMDB_sentiment_analysis.txt*

#### Hàm mất mát BCELoss + Sigmoid

- **Là gì:** Binary Cross-Entropy Loss cho phân loại nhị phân, dùng chung với Sigmoid ở đầu ra (xác suất trong [0,1]).
- **Ý nghĩa & mục đích:** Bài phân loại 2 lớp (chuỗi hợp lệ/không, phim tích cực/tiêu cực). Sigmoid ép output về xác suất, BCELoss so với nhãn 0/1.
- **Code:**

```python
criterion = nn.BCELoss()
# output đã qua sigmoid; targets phải là float
loss = criterion(output, targets.float())
```
- **Visual:** Vẽ đường loss và val_loss theo epoch để theo dõi hội tụ. Không có hình riêng cho hàm loss.
- **Lưu ý:** BCELoss cần input ĐÃ qua Sigmoid (trong [0,1]); nếu đưa logit thô sẽ sai. targets phải .float() nếu không lỗi kiểu. Khuyến nghị hiện đại: bỏ Sigmoid ở model và dùng nn.BCEWithLogitsLoss (gộp sigmoid + BCE, ổn định số học hơn, tránh log(0)).
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 3 - IMDB_sentiment_analysis.txt*

#### Bộ tối ưu Adam / RMSprop

- **Là gì:** Thuật toán cập nhật trọng số theo gradient với learning rate thích nghi. Khoá dùng Adam (bài nhớ), RMSprop (bài ngoặc và IMDB).
- **Ý nghĩa & mục đích:** Cập nhật tham số để giảm loss. Adam/RMSprop điều chỉnh bước học riêng cho từng tham số, hội tụ nhanh hơn SGD thường trên RNN.
- **Code:**

```python
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# hoặc (IMDB):
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
```
- **Visual:** Không có hình. Có thể vẽ đường loss với vài lr khác nhau để thấy lr quá cao gây dao động/phân kỳ.
- **Lưu ý:** lr quá cao -> loss dao động/phân kỳ; quá thấp -> học chậm. Bài ngoặc dùng RMSprop KHÔNG momentum; chỉ IMDB thêm momentum=0.9. Phải truyền model.parameters() ĐÚNG mô hình đang train (IMDB tạo optimizer riêng cho model_rnn và model_lstm).
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt; 3 - IMDB_sentiment_analysis.txt*

#### Vòng lặp huấn luyện RNN (zero_grad / backward / step)

- **Là gì:** Chu trình chuẩn mỗi batch: xoá gradient cũ, chạy forward, tính loss, backward tính gradient, step cập nhật trọng số.
- **Ý nghĩa & mục đích:** Khuôn train tái sử dụng cho mọi mô hình chuỗi. Với LSTM giữ state còn detach hidden mỗi batch và truyền hidden qua forward.
- **Code:**

```python
model.train()
for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    h = tuple(e.data for e in h)   # detach hidden (chỉ với model giữ state)
    model.zero_grad()
    output, h = model(inputs, h)
    loss = criterion(output, targets.float())
    loss.backward()
    optimizer.step()
```
- **Visual:** Không có hình cho vòng lặp. Kết quả in ra loss/accuracy mỗi epoch để theo dõi.
- **Lưu ý:** QUÊN zero_grad -> gradient cộng dồn qua các batch, học sai. Thứ tự phải là zero_grad -> forward -> backward -> step. 'running_loss += loss' giữ TENSOR còn gắn đồ thị (tốn RAM và giữ graph) — nên cộng loss.item(). Ở IMDB model.forward chỉ nhận inputs (không hidden), nên bỏ dòng detach.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 3 - IMDB_sentiment_analysis.txt*

#### Hàm tính độ chính xác nhị phân (accuracy)

- **Là gì:** Chuyển output xác suất thành nhãn 0/1 bằng ngưỡng 0.5 rồi so với nhãn thật, trả về tỉ lệ đúng.
- **Ý nghĩa & mục đích:** Đo hiệu năng dễ hiểu hơn loss. Dùng cho cả train và validation mỗi epoch.
- **Code:**

```python
def accuracy(outputs, targets):
    labels = torch.zeros(len(outputs)).to(device)
    ones_index = torch.where(outputs > 0.5)[0]
    labels[ones_index] = 1.0
    return torch.sum(labels == targets.float()) / float(len(labels))
```
- **Visual:** Vẽ đường accuracy & val_accuracy theo epoch (xem thẻ vẽ history).
- **Lưu ý:** Ngưỡng 0.5 chỉ hợp khi output là xác suất qua Sigmoid. Với dữ liệu MẤT CÂN BẰNG nhãn (val của IMDB bị lệch do cắt tuần tự), accuracy có thể gây hiểu lầm — cần cân bằng tập hoặc dùng thêm chỉ số khác (F1). Trả về tensor nên .item() khi cộng dồn.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 3 - IMDB_sentiment_analysis.txt*

#### Lưu lịch sử & vẽ đường học (history + plot)

- **Là gì:** Dict lưu loss/accuracy của train và validation qua từng epoch, rồi vẽ 2 biểu đồ (accuracy và loss) so sánh train vs val.
- **Ý nghĩa & mục đích:** Theo dõi quá trình học: phát hiện overfitting (train tốt lên nhưng val xấu đi) và biết khi nào nên dừng.
- **Code:**

```python
history = {'loss':[], 'val_loss':[], 'accuracy':[], 'val_accuracy':[]}
# cuối mỗi epoch: history['loss'].append(running_loss)  # running_loss đã .item()/num_batches
epochs = range(1, len(history['accuracy'])+1)
plt.plot(epochs, history['accuracy'], 'bo', label='Training acc')
plt.plot(epochs, history['val_accuracy'], 'b', label='Validation acc')
plt.legend(); plt.figure()
plt.plot(epochs, history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history['val_loss'], 'b', label='Validation loss')
plt.legend(); plt.show()
```
- **Visual:** 2 hình: (1) accuracy train (chấm 'bo') vs val (đường 'b'); (2) loss tương tự. plt.ylim(0,1) cho accuracy. Khoảng cách train-val nới rộng = overfitting.
- **Lưu ý:** running_loss/accuracy phải .item() và chia num_batches TRƯỚC khi append, nếu không lưu tensor kèm đồ thị -> tốn RAM. 'bo' = chấm xanh (train), 'b' = đường xanh (val).
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 3 - IMDB_sentiment_analysis.txt*

#### Trực quan hoá output LSTM bằng chuyển trọng số (model_viz + load_state_dict)

- **Là gì:** Tạo một mô hình phụ chỉ gồm Embedding + LSTM (không có lớp phân loại), nạp trọng số từ mô hình đã train, để lấy output LSTM ở MỌI bước thời gian rồi vẽ heatmap.
- **Ý nghĩa & mục đích:** Mô hình gốc chỉ trả 1 nhãn; muốn 'nhìn vào trong' xem LSTM tính gì ở từng ký tự thì cần model phụ xuất full chuỗi. Giúp hiểu neuron nào phản ứng với ký tự quan trọng.
- **Code:**

```python
from collections import OrderedDict
model_viz = nn.Sequential(OrderedDict({
    'embedding': nn.Embedding.from_pretrained(embedding_matrix, freeze=True),
    'lstm': nn.LSTM(embedding_dim, hidden_dim, 1, batch_first=True)})).to(device)
model_viz.embedding.load_state_dict(model.embedding.state_dict())
model_viz.lstm.load_state_dict(model.lstm.state_dict())
lstm_outputs, h = model_viz(viz_vector)
plt.imshow(lstm_outputs[0,:,:].cpu().detach().numpy().transpose()); plt.colorbar()
```
- **Visual:** Heatmap: hàng = 10 neuron hidden, cột = từng ký tự trong chuỗi; màu = giá trị activation. Đổi 1 ký tự (quan trọng vs không) rồi so heatmap để thấy neuron nào thay đổi.
- **Lưu ý:** load_state_dict yêu cầu kiến trúc lớp con KHỚP hệt (cùng dim). model_viz KHÔNG train, chỉ mượn trọng số. nn.Sequential khiến model_viz(x) trả thẳng (output, hidden) của LSTM (không có lớp phân loại). Vì LSTM chia sẻ trọng số qua mọi bước nên chạy được với chuỗi dài hơn lúc train, nhưng quá dài mô hình gốc có thể sai.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt*

#### Tổng quát hoá sang độ dài chuỗi khác (generalisation)

- **Là gì:** LSTM huấn luyện trên chuỗi độ dài cố định vẫn chạy được và cho kết quả đúng trên chuỗi dài hơn / lồng sâu hơn khi đưa vào model_viz.
- **Ý nghĩa & mục đích:** Chứng tỏ LSTM học được QUY LUẬT chuỗi (nhớ ký tự, khớp ngoặc) chứ không chỉ thuộc lòng độ dài cụ thể. Bài ngoặc tổng quát hoá tốt hơn bài nhớ.
- **Code:**

```python
# train ở pad_length=10 nhưng thử chuỗi dài hơn:
viz_string = 'abababRabbbbbSababab'   # dài 20
viz_vector = string_to_int_vec(viz_string, len(viz_string), c2i)
viz_vector = torch.tensor(viz_vector.reshape([1,-1])).type(torch.int64).to(device)
lstm_outputs, h = model_viz(viz_vector)
```
- **Visual:** Vẽ heatmap output cho chuỗi dài hơn train và xem neuron nhớ vẫn hoạt động đúng ở vị trí R/S dù xa hơn.
- **Lưu ý:** Có giới hạn: 'chuỗi dài tới đâu thì mô hình gốc còn đúng?' — quá dài sẽ hỏng (câu hỏi mở trong source). Vì LSTM chia sẻ trọng số qua mọi bước nên input độ dài khác nhau vẫn chạy được. Với model_viz reshape phải ép kiểu int64 trước khi vào Embedding.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt*

#### Kiểm tra trọng số lớp cuối (model.fc.weight)

- **Là gì:** In các trọng số của lớp Linear cuối cùng để xem output LSTM nào đóng góp vào quyết định phân loại.
- **Ý nghĩa & mục đích:** Diễn giải mô hình: nếu không trọng số nào gần 0 nghĩa là mọi neuron output của LSTM đều được dùng để phân loại chuỗi.
- **Code:**

```python
for parameter in model.fc.parameters():
    print(parameter)
# hoặc trực tiếp:
model.fc.weight
```
- **Visual:** Vẽ bar chart độ lớn các trọng số của fc. Bài tập gợi ý: nhân ma trận output model_viz với trọng số fc để xem 'điểm phân loại' tại mỗi ký tự.
- **Lưu ý:** Không trọng số nào gần 0 -> tất cả neuron hidden đều quan trọng, khó cắt tỉa. Đây chỉ là diễn giải thô, không phải giải thích nhân quả đầy đủ. Ở IMDB lớp cuối tên là model.hidden chứ không phải model.fc.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 2 - What_can_RNN_learn_with_brackets.txt*

#### Mã hoá review tự viết để thử mô hình (encode_my_reviews)

- **Là gì:** Hàm biến các câu review do người dùng tự viết thành tensor id đã token hoá và pad về maxlen, để đưa vào mô hình dự đoán.
- **Ý nghĩa & mục đích:** Cho phép thử mô hình bằng câu tự nghĩ ra ('phim dở nhất', 'phim hay nhất') để xem nó thực sự hiểu tới đâu, và thử 'đánh lừa' nó.
- **Code:**

```python
maxlen = 500
def encode_my_reviews(list_of_my_reviews):
    my_sequences = [torch.tensor(vocab(tokenizer(r)[0:maxlen])) for r in list_of_my_reviews]
    my_data = pad_sequence(my_sequences, True, padding_value=0)
    tmp = torch.zeros(my_data.shape[0], maxlen).type(torch.int64)
    for i, d in enumerate(my_data):
        tmp[i, 0:d.shape[0]] = my_data[i, 0:d.shape[0]]
    return tmp
# dùng: model(encode_my_reviews(my_reviews).to(device))
```
- **Visual:** In tensor mã hoá + xác suất dự đoán cho từng câu. So sánh dự đoán của RNN (kém) vs LSTM (tốt) trên cùng câu.
- **Lưu ý:** Phải pad thủ công về đúng maxlen=500 (tạo tmp) vì pad_sequence chỉ pad tới câu dài nhất trong lô, còn model mong độ dài cố định. tokenizer/vocab dùng như biến TOÀN CỤC — phải nạp đúng bộ đã train. Từ ngoài vocab thành <unk>, có thể làm mô hình hiểu sai.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Chọn thiết bị CPU/GPU (device)

- **Là gì:** Đoạn kiểm tra có CUDA (GPU) hay không rồi đặt device tương ứng; sau đó .to(device) cho model và dữ liệu.
- **Ý nghĩa & mục đích:** Huấn luyện RNN trên chuỗi dài (IMDB 500 từ) nhanh hơn nhiều trên GPU. Viết code chạy được cả hai nơi mà không sửa.
- **Code:**

```python
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
model = model.to(device)
inputs = inputs.to(device, non_blocking=True)
```
- **Visual:** In device ra ('cuda:0' hoặc 'cpu'). Không có hình.
- **Lưu ý:** Model VÀ dữ liệu phải cùng device, nếu lệch -> RuntimeError. Khi lấy kết quả để vẽ/in phải .cpu().detach().numpy(). non_blocking=True chỉ có tác dụng thật với pinned memory. Cách hiện đại gọn hơn: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu').
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 3 - IMDB_sentiment_analysis.txt*

#### Lưu/nạp dữ liệu đã tiền xử lý (pickle)

- **Là gì:** Dùng pickle để lưu (serialize) toàn bộ biến đã tiền xử lý (texts, vocab, x_train...) ra file, rồi nạp lại nhanh cho lần sau.
- **Ý nghĩa & mục đích:** Tiền xử lý IMDB (đọc 25000 file, token hoá) rất chậm; pickle giúp làm một lần rồi tái dùng, khỏi chạy lại.
- **Code:**

```python
import pickle
pickle.dump((texts, labels, vocab, x_train, y_train, x_val, y_val, tokenizer),
            open("imdb_raw_and_coded_data.pickle", "wb"))
# lần sau:
(texts, labels, vocab, x_train, y_train, x_val, y_val, tokenizer) = \
    pickle.load(open("imdb_raw_and_coded_data.pickle", "rb"))
```
- **Visual:** Không có hình. Kiểm tra bằng cách in shape x_train sau khi nạp.
- **Lưu ý:** File pickle có thể KHÔNG nạp được nếu đổi phiên bản Python/torchtext (vocab, tokenizer là object torchtext) -> phải xử lý lại từ dữ liệu thô. Pickle KHÔNG an toàn khi nạp file từ nguồn lạ (có thể chạy mã tùy ý). Nhớ 'wb'/'rb' (binary). Với riêng model nên lưu state_dict thay vì pickle cả object.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Kiểm tra cân bằng nhãn (plt.hist labels)

- **Là gì:** Vẽ histogram phân bố nhãn 0/1 của tập train và validation để xem hai lớp có cân bằng không.
- **Ý nghĩa & mục đích:** Nhãn lệch làm accuracy dễ gây hiểu lầm và mô hình thiên về lớp đa số. Bài IMDB cắt 20000/5000 tuần tự khiến val bị lệch nhãn — cần biết trước.
- **Code:**

```python
plt.hist(labels)
plt.hist(y_train.numpy())
plt.hist(y_val.numpy())
```
- **Visual:** Histogram 2 cột (nhãn 0 và 1). Nếu 2 cột chênh nhiều -> mất cân bằng.
- **Lưu ý:** Chia tuần tự (không xáo trước) làm val có thể toàn 1 lớp -> val accuracy vô nghĩa. Khắc phục: xáo trộn TRƯỚC khi chia hoặc tạo tập cân bằng (chính là bài tập cuối notebook IMDB). y_train là tensor nên cần .numpy() để plt.hist.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*

#### Chế độ train/eval và tắt gradient khi đánh giá (model.eval / torch.no_grad)

- **Là gì:** model.train() bật chế độ huấn luyện, model.eval() bật chế độ đánh giá; torch.no_grad() tắt việc dựng đồ thị autograd trong vòng validation/dự đoán.
- **Ý nghĩa & mục đích:** Tách rõ pha train và pha đánh giá. eval() làm dropout/batchnorm hành xử đúng lúc test; no_grad() khi tính val_loss/dự đoán giúp tiết kiệm bộ nhớ và chạy nhanh hơn vì không cần gradient.
- **Code:**

```python
model.train()
# ... vòng lặp train ...
model.eval()
with torch.no_grad():
    for test_inputs, test_labels in test_loader:
        test_outputs = model(test_inputs.to(device))
        val_loss += criterion(test_outputs, test_labels.float().to(device))
```
- **Visual:** Không có hình. Có thể in ra so sánh thời gian/bộ nhớ khi có và không có no_grad trên vòng validation.
- **Lưu ý:** Notebook trong khoá CHỈ gọi model.train() mà THIẾU model.eval() và torch.no_grad() ở vòng test — với mô hình có dropout/batchnorm sẽ cho val sai, và không no_grad() thì lãng phí RAM (giữ đồ thị). Nhớ gọi lại model.train() trước epoch train kế tiếp.
- *Nguồn: 1 - What_can_recurrent_nets_learn.txt; 3 - IMDB_sentiment_analysis.txt*

#### GRU và LSTM hai chiều (nn.GRU / bidirectional)

- **Là gì:** GRU (Gated Recurrent Unit) là biến thể hồi tiếp có cổng nhẹ hơn LSTM (ít tham số). bidirectional=True cho RNN/LSTM/GRU đọc chuỗi theo cả hai chiều rồi ghép hidden.
- **Ý nghĩa & mục đích:** Là các mở rộng tự nhiên của mô hình chuỗi để thử cải thiện IMDB (chính là bài tập cuối notebook): GRU train nhanh, đôi khi tốt ngang LSTM; bidirectional cho mỗi bước thấy được cả ngữ cảnh trước và sau, hữu ích khi cả câu đã có sẵn.
- **Code:**

```python
self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
# hoặc LSTM hai chiều:
self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
# khi bidirectional: output có kích thước hidden_dim*2
self.hidden = nn.Linear(hidden_dim*2, num_classes)
```
- **Visual:** Vẽ 3 đường val_accuracy (RNN, LSTM, GRU) hoặc so LSTM 1 chiều vs 2 chiều để thấy khác biệt.
- **Lưu ý:** bidirectional=True làm chiều output của RNN gấp ĐÔI (hidden_dim*2) -> lớp Linear phía sau phải nhận hidden_dim*2, quên là lỗi shape. GRU trả về (output, h_n) 1 tensor giống RNN (không phải tuple như LSTM). Đây là gợi ý ở bài tập, không có sẵn trong code khoá.
- *Nguồn: 3 - IMDB_sentiment_analysis.txt*


---

### 8.8 Metric Learning & GAN

#### Học biểu diễn / hàm nhúng (Embedding function / Metric learning)

- **Là gì:** Một mạng biến mỗi ảnh thành một vector nhúng (embedding). Mục tiêu không phải phân lớp, mà là làm cho khoảng cách giữa các vector phản ánh độ giống nhau: cùng loại thì gần, khác loại thì xa.
- **Ý nghĩa & mục đích:** Dùng cho one-shot learning: khi mỗi lớp chỉ có rất ít mẫu (thậm chí 1), ta không huấn luyện bộ phân lớp cho từng lớp mà học một hàm 'đo độ giống'. Sau đó phân loại vật mới bằng cách so khoảng cách embedding với vài mẫu tham chiếu.
- **Code:**

```python
class embedding_model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 30), nn.ReLU(),
            nn.Linear(30, 30), nn.Tanh())
    def forward(self, inputs):
        return self.model(inputs)
```
- **Visual:** Vẽ histogram khoảng cách của các cặp 'same' và 'different' chồng lên nhau (plt.hist(..., alpha=0.5)). Embedding tốt sẽ tách hai đám: same dồn về gần 0, different dồn ra xa.
- **Lưu ý:** Đây KHÔNG phải bài phân lớp — output không phải xác suất lớp mà là vector. Lớp cuối dùng Tanh để chặn embedding trong [-1,1], tránh vector nổ to. Deprecation nhẹ: viết super().__init__() thay cho super(embedding_model_1, self).__init__() kiểu Python 2.
- *Nguồn: MNIST_Siamese.txt*

#### Mạng Siamese (Siamese Network) & chia sẻ trọng số (weight sharing)

- **Là gì:** Mạng nhận MỘT cặp ảnh, đẩy cả hai qua CÙNG một embedding model (dùng chung trọng số), rồi trả về khoảng cách Euclid giữa hai vector nhúng.
- **Ý nghĩa & mục đích:** Vì hai nhánh dùng chung trọng số nên cùng một ảnh luôn cho cùng một embedding, bất kể nằm ở nhánh trái hay phải. Đó là cách 'nhúng một lần, so sánh cặp' để huấn luyện theo khoảng cách thay vì theo nhãn lớp.
- **Code:**

```python
class SiameseNetworkModel(nn.Module):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model  # dùng chung 1 model
    def forward(self, inputs):
        out_a = self.embedding_model(inputs[:, 0, :, :])
        out_b = self.embedding_model(inputs[:, 1, :, :])
        out_a_b = torch.stack([out_a, out_b]).transpose(0, 1)
        return euclidean_distances_of_rows(out_a_b)
```
- **Visual:** Sơ đồ hình chữ Y ngược: hai ảnh -> cùng một khối embedding (share weights) -> hai vector -> hộp tính khoảng cách -> một số vô hướng d.
- **Lưu ý:** Chia sẻ trọng số đạt được bằng cách gọi CÙNG một self.embedding_model hai lần, KHÔNG phải tạo hai model riêng. Tạo hai model là sai (mỗi bên học khác nhau).
- *Nguồn: MNIST_Siamese.txt*

#### Hàm mất mát tương phản (Contrastive loss)

- **Là gì:** Loss cho cặp: nếu cặp giống nhau (y=0) thì phạt theo d^2 (kéo lại gần); nếu khác nhau (y=1) thì phạt theo max(1-d,0)^2 (đẩy ra xa ít nhất bằng lề margin=1).
- **Ý nghĩa & mục đích:** Thay cho log-loss/cross-entropy quen thuộc. Nó ép embedding của cặp giống nhau tiến về 0 khoảng cách, và cặp khác nhau tách ra ít nhất 1 đơn vị. Khi đã đủ xa (d>=1) thì không phạt nữa, nên mạng không phí sức đẩy vô hạn.
- **Code:**

```python
def contrastive_loss(y, d):
    zero_tensor = torch.zeros(d.shape)
    loss = (((1 - y) * torch.square(d))
            + y * torch.square(torch.max(1 - d, zero_tensor))).mean()
    return loss
```
- **Visual:** Vẽ hai nhánh loss theo d: nhánh same (y=0) là parabol d^2 tăng dần; nhánh different (y=1) là (1-d)^2 giảm về 0 tại d=1 rồi phẳng. Cho thấy 'margin' = 1.
- **Lưu ý:** Quy ước nhãn ở đây ngược trực giác: y=0 là GIỐNG, y=1 là KHÁC. Margin cứng bằng 1 gắn liền với embedding cỡ [-1,1] của Tanh — đổi thang embedding thì nên xem lại margin. Thứ tự tham số là (y, d): khi gọi phải truyền nhãn trước, khoảng cách sau.
- *Nguồn: MNIST_Siamese.txt*

#### Khoảng cách Euclid theo hàng (Euclidean distance of rows)

- **Là gì:** Hàm nhận batch gồm các cặp vector, tính khoảng cách Euclid giữa vector 0 và vector 1 của từng dòng: sqrt(sum((a-b)^2)).
- **Ý nghĩa & mục đích:** Là 'thước đo' mà Siamese network xuất ra và contrastive loss ăn vào. Giữ keepdim để giữ đúng shape cột (n,1) khớp với nhãn y.
- **Code:**

```python
def euclidean_distances_of_rows(inputs):
    assert inputs.shape[1] == 2
    diff = inputs[:, 0, :] - inputs[:, 1, :]
    return torch.sqrt(torch.sum(torch.square(diff), dim=1, keepdim=True))
```
- **Visual:** Kiểm tra bằng scatter: plt.plot(distances_network, distances_numpy, '.') — nếu mạng tính đúng thì các điểm nằm trên đường thẳng y=x (hàm đồng nhất).
- **Lưu ý:** Tránh cặp hai ảnh y hệt nhau (d=0) vì đạo hàm của sqrt tại 0 phân kỳ khi backprop; vì thế lúc tạo cặp có kiểm tra x1 != x2. Tên tham số chuẩn trong PyTorch là keepdim (source viết keepdims — vẫn chạy như một alias tương thích numpy, nhưng nên dùng keepdim).
- *Nguồn: MNIST_Siamese.txt*

#### Tạo tập cặp cho Siamese (Constructing pairs dataset)

- **Là gì:** Sinh dữ liệu huấn luyện dạng cặp: mỗi mẫu là (ảnh1, ảnh2) kèm nhãn y = 0 nếu cùng chữ số, 1 nếu khác. Có cân bằng để số cặp giống ≈ số cặp khác.
- **Ý nghĩa & mục đích:** Metric learning cần dữ liệu là các CẶP, không phải ảnh đơn. Với 10 chữ số, xác suất trùng ngẫu nhiên chỉ 1/10, nên chấp nhận cặp khác với xác suất ~1/9 để hai loại cân bằng.
- **Code:**

```python
def construct_pairs_dataset(n_data, images, labels):
    x_train1 = torch.zeros((n_data, 28, 28, 1))
    x_train2 = torch.zeros((n_data, 28, 28, 1))
    y_train = torch.zeros([n_data, 1])
    for i in range(n_data):
        while True:
            while True:
                x1 = randint(0, n_data - 1); x2 = randint(0, n_data - 1)
                if x1 != x2: break
            if labels[x1] == labels[x2]:
                y_train[i, 0] = 0; break
            elif np.random.random() < 0.111111111:
                y_train[i, 0] = 1; break
        x_train1[i, :, :, 0] = images[x1]; x_train2[i, :, :, 0] = images[x2]
    return ([x_train1, x_train2], y_train)
```
- **Visual:** Chọn idx ngẫu nhiên, plt.subplots(1,2) hiện hai ảnh của cặp và in py_train[idx].item() để mắt thường xác nhận cùng số -> 0, khác số -> 1.
- **Lưu ý:** Nhãn không cân bằng nếu bỏ bước lọc xác suất 1/9 (sẽ toàn cặp khác). Sau khi torch.stack, phải permute(1,0,4,2,3) để đưa channel về đúng chiều cho conv/Flatten.
- *Nguồn: MNIST_Siamese.txt*

#### Embedding bằng CNN cho Siamese (Convolutional embedding model)

- **Là gì:** Biến thể embedding dùng các lớp Conv2d + ReLU + MaxPool rồi Flatten + Linear + Tanh, thay cho MLP thuần.
- **Ý nghĩa & mục đích:** CNN khai thác cấu trúc không gian của ảnh nên thường cho embedding tách 'same/different' tốt hơn MLP. Dùng để so sánh: 'một số mạng nhúng tốt hơn hẳn số khác'.
- **Code:**

```python
class embedding_model_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 20, 3), nn.ReLU(),
            nn.Conv2d(20, 40, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(40, 40, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(), nn.Linear(1000, 30), nn.Tanh())
    def forward(self, inputs):
        return self.model(inputs)
```
- **Visual:** So sánh hai histogram same/different giữa model MLP và model CNN trên TẬP TEST: CNN thường tách rõ hơn, ít chồng lấn hơn.
- **Lưu ý:** Số 1000 ở Linear(1000,30) phụ thuộc kích thước ảnh sau chuỗi conv/pool — đổi kernel/pool là phải tính lại, nếu không sai shape. Kiểm tra nhanh bằng embedding_model(x).shape.
- *Nguồn: MNIST_Siamese.txt*

#### Đánh giá embedding bằng histogram khoảng cách (Same/different distance histogram)

- **Là gì:** Sau huấn luyện, tính khoảng cách dự đoán cho toàn bộ cặp, rồi vẽ histogram riêng cho cặp same (y=0) và cặp different (y=1), làm trên cả train và test.
- **Ý nghĩa & mục đích:** Đây là 'thước đo chất lượng' của metric learning: nếu hai đám tách rời -> embedding tốt. Nếu trên train tách nhưng trên test chồng lấn nhiều -> đang overfit.
- **Code:**

```python
preds = siamese_network_model(pairs_train_tensor).detach().reshape(-1)
plt.hist(preds[py_train[:, 0] == 0].numpy(), alpha=0.5, label='same')
plt.hist(preds[py_train[:, 0] == 1].numpy(), alpha=0.5, label='different')
# so trung binh:
torch.mean(preds[py_train[:, 0] == 0]), torch.mean(preds[py_train[:, 0] == 1])
```
- **Visual:** Hai histogram chồng nhau (alpha=0.5). Kỳ vọng: đám 'same' dồn về khoảng cách nhỏ, đám 'different' dồn về khoảng cách lớn; khoảng hở giữa hai đám càng rộng càng tốt.
- **Lưu ý:** Luôn .detach() trước khi .numpy() để tách khỏi đồ thị tính đạo hàm. So sánh trung bình same vs different chỉ là tóm tắt — vẫn nên nhìn histogram vì độ chồng lấn mới là thứ quan trọng.
- *Nguồn: MNIST_Siamese.txt*

#### Vòng lặp huấn luyện Siamese từ tensor cặp (Siamese training loop + TensorDataset)

- **Là gì:** Đóng gói tensor cặp và nhãn thành TensorDataset -> DataLoader, rồi chạy vòng lặp huấn luyện chuẩn: zero_grad -> forward -> loss -> backward -> step.
- **Ý nghĩa & mục đích:** Metric learning vẫn train như mọi mạng khác, chỉ khác ở chỗ input là cặp và loss là contrastive. Đây là mẫu vòng lặp tái sử dụng được cho bất kỳ mạng nào nhận (inputs, labels) từ DataLoader.
- **Code:**

```python
dataset = torch.utils.data.TensorDataset(pairs_train_tensor, py_train)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
optimizer = optim.RMSprop(siamese_network_model.parameters(), lr=0.001)

def train_model(model, num_epochs, trainloader, optimizer, loss_criterion):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(labels, outputs)   # (y, d)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('epoch %d loss %.3f' % (epoch, running_loss / len(trainloader)))
```
- **Visual:** In loss trung bình mỗi epoch; nếu loss giảm dần và histogram same/different tách ra thì vòng lặp chạy đúng.
- **Lưu ý:** loss_criterion nhận (labels, outputs) = (y, d) đúng thứ tự của contrastive_loss — đảo là sai. Ở đây dùng RMSprop (không phải Adam) như bản gốc; TensorDataset yêu cầu tensor đã stack + permute đúng shape trước khi tạo loader.
- *Nguồn: MNIST_Siamese.txt*

#### GAN — mạng đối kháng sinh mẫu (Generative Adversarial Network)

- **Là gì:** Hai mạng đấu nhau: Generator sinh ảnh giả từ nhiễu, Discriminator cố phân biệt thật/giả. Huấn luyện xen kẽ cho tới khi ảnh giả đủ giống thật.
- **Ý nghĩa & mục đích:** Để học phân phối dữ liệu và SINH mẫu mới (ví dụ ảnh chữ số MNIST) mà không cần công thức phân phối. D làm 'thầy chấm', G học cách đánh lừa thầy.
- **Code:**

```python
conditional = False
discriminator = DiscriminatorModel().to(device)
generator = GeneratorModel().to(device)
loss = nn.BCELoss()
(generator, discriminator, G_loss, D_loss) = train(
    conditional, discriminator, generator, loss,
    data_loader, device, batch_size)
```
- **Visual:** Sau train, sinh nhiễu và vẽ lưới ảnh: noise=torch.randn(24,100); plot_images(generator(noise).view(-1,28,28)). Nhìn xem có ra hình chữ số không.
- **Lưu ý:** conditional là tham số VỊ TRÍ đầu tiên của train() — phải gán conditional=False thành một dòng riêng rồi truyền positional; viết train(conditional=False, discriminator, ...) là lỗi cú pháp (positional sau keyword). GAN dễ mất ổn định: loss của G và D dao động là bình thường, đừng kỳ vọng loss giảm mượt như bài phân lớp.
- *Nguồn: CGANs.txt*

#### GAN có điều kiện (Conditional GAN - CGAN)

- **Là gì:** GAN được thêm thông tin nhãn lớp vào cả Generator lẫn Discriminator, để có thể YÊU CẦU sinh đúng một lớp cụ thể (ví dụ 'hãy sinh chữ số 7').
- **Ý nghĩa & mục đích:** GAN thường sinh mẫu ngẫu nhiên không kiểm soát lớp. CGAN cho phép điều khiển: đưa label vào thì G sinh đúng lớp đó, D chấm 'thật/giả CÓ khớp nhãn không'.
- **Code:**

```python
conditional = True
c_discriminator = DiscriminatorModel(conditional).to(device)
c_generator = GeneratorModel(conditional).to(device)
loss = nn.BCELoss()
(c_generator, c_discriminator, G_loss, D_loss) = train(
    conditional, c_discriminator, c_generator, loss,
    data_loader, device, batch_size)
# sinh theo nhãn chỉ định:
fake_labels = torch.randint(0, 10, (num_imgs,)).to(device)
generated = c_generator(noise, fake_labels)
```
- **Visual:** Vẽ lưới ảnh sinh ra với title = nhãn yêu cầu (fake_labels). Nếu CGAN học tốt, ảnh trong ô có title '7' phải trông giống số 7.
- **Lưu ý:** Cùng một class GeneratorModel/DiscriminatorModel dùng cho cả hai chế độ nhờ cờ conditional; khi conditional=True phải TRUYỀN labels vào forward, không thì phần embedding nhãn bị bỏ qua.
- *Nguồn: CGANs.txt*

#### Mạng sinh (Generator model)

- **Là gì:** Mạng FC nhận vector nhiễu 100 chiều (+ embedding nhãn nếu conditional), tăng dần kích thước qua các Linear (256->512->784) và xuất ảnh 784 pixel, squash bằng Tanh về [-1,1].
- **Ý nghĩa & mục đích:** Biến nhiễu ngẫu nhiên thành ảnh trông như thật. Các lớp Linear tăng chiều đóng vai 'upsample' độ phân giải. Tanh ở cuối để khớp với dữ liệu thật đã chuẩn hoá về [-1,1].
- **Code:**

```python
class GeneratorModel(nn.Module):
    def __init__(self, conditional=False):
        super().__init__()
        input_dim = 100 + (10 if conditional else 0)
        self.label_embedding = nn.Embedding(10, 10)
        self.hidden_layer1 = nn.Sequential(nn.Linear(input_dim, 256), nn.LeakyReLU(0.2))
        self.hidden_layer2 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2))
        self.output_layer = nn.Sequential(nn.Linear(512, 784), nn.Tanh())
    def forward(self, x, labels=None):
        if labels is not None:
            x = torch.cat([x, self.label_embedding(labels)], 1)
        return self.output_layer(self.hidden_layer2(self.hidden_layer1(x)))
```
- **Visual:** Sơ đồ ống loe dần: nhiễu 100 -> 256 -> 512 -> 784, rồi reshape 28x28 thành ảnh. Có thể vẽ ảnh output ở vài epoch để thấy nó rõ dần.
- **Lưu ý:** Tanh output [-1,1] BẮT BUỘC khớp với transform Normalize(0.5,0.5) của dữ liệu thật. Nếu dữ liệu thật để [0,1] mà G xuất [-1,1] thì D phân biệt được ngay -> G không học nổi. (Bản gốc thêm .to(device) ở cuối forward — thừa nếu model và input đã ở đúng device.)
- *Nguồn: CGANs.txt*

#### Mạng phân biệt (Discriminator model)

- **Là gì:** Mạng FC nhận ảnh 784 pixel (+ embedding nhãn nếu conditional), giảm chiều (512->256->1) và xuất MỘT xác suất qua Sigmoid: khả năng ảnh là thật.
- **Ý nghĩa & mục đích:** Đóng vai 'giám khảo' phân biệt thật/giả. Output 1 số trong [0,1] hợp với BCELoss. Có Dropout để chống overfit và tránh D quá mạnh nuốt chửng G.
- **Code:**

```python
class DiscriminatorModel(nn.Module):
    def __init__(self, conditional=False):
        super().__init__()
        input_dim = 784 + (10 if conditional else 0)
        self.label_embedding = nn.Embedding(10, 10)
        self.hidden_layer1 = nn.Sequential(nn.Linear(input_dim, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.hidden_layer2 = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.output_layer = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
    def forward(self, x, labels=None):
        if labels is not None:
            x = torch.cat([x, self.label_embedding(labels)], 1)
        return self.output_layer(self.hidden_layer2(self.hidden_layer1(x)))
```
- **Visual:** Sơ đồ ống thu hẹp: 784 -> 512 -> 256 -> 1 (Sigmoid). Có thể vẽ phân bố output D cho ảnh thật vs ảnh giả — khi cân bằng, cả hai xúm quanh 0.5.
- **Lưu ý:** Output cuối là Sigmoid ra [0,1] để dùng với nn.BCELoss. Modern PyTorch khuyên bỏ Sigmoid và dùng nn.BCEWithLogitsLoss (ổn định số học hơn); nếu vẫn dùng Sigmoid + BCELoss thì cẩn thận không đặt sigmoid hai lần.
- *Nguồn: CGANs.txt*

#### Nhúng nhãn để điều kiện hoá (Label embedding với nn.Embedding)

- **Là gì:** Dùng nn.Embedding(10, 10) biến nhãn lớp (0..9) thành vector 10 chiều học được, rồi torch.cat vào đầu vào của G và D.
- **Ý nghĩa & mục đích:** Cách đưa thông tin lớp vào mạng. Tài liệu nói embedding học được 'tốt hơn one-hot đơn giản' vì mạng tự học biểu diễn nhãn có ý nghĩa thay vì vector cứng 0/1.
- **Code:**

```python
self.label_embedding = nn.Embedding(10, embedding_size)  # 10 lop -> vector 10 chieu
# trong forward:
if labels is not None:
    c = self.label_embedding(labels)
    x = torch.cat([x, c], 1)   # noi nhan vao dau vao
```
- **Visual:** Có thể trực quan bằng cách giảm chiều (PCA/t-SNE) 10 vector embedding của 10 chữ số để xem chúng tách nhau ra sao sau khi học.
- **Lưu ý:** nn.Embedding nhận CHỈ SỐ nguyên (LongTensor), không phải one-hot. Nối bằng torch.cat theo dim=1 nên phải khớp batch dimension; sai dim là lỗi shape.
- *Nguồn: CGANs.txt*

#### Vòng huấn luyện đối kháng: bước D rồi bước G (Adversarial training loop)

- **Là gì:** Mỗi minibatch: (1) cập nhật D trên ảnh thật (nhãn 1) và ảnh giả (nhãn 0); (2) cập nhật G bằng cách ép D chấm ảnh giả là 'thật' (nhãn 1). Hai optimizer riêng.
- **Ý nghĩa & mục đích:** D và G có mục tiêu ngược nhau nên phải update tách biệt, dùng hai optimizer Adam riêng. D học phân biệt; G học đánh lừa D. Đây là 'trò chơi' cốt lõi của GAN.
- **Code:**

```python
d_opt = optim.Adam(discriminator.parameters(), lr=0.0002)
g_opt = optim.Adam(generator.parameters(), lr=0.0002)
# --- D step ---
d_opt.zero_grad()
real_loss = loss(discriminator(true_data, digit_labels).view(bs), ones)
fake_loss = loss(discriminator(generated_data.detach(), fake_labels).view(bs), zeros)
d_loss = (real_loss + fake_loss) / 2
d_loss.backward(); d_opt.step()
# --- G step ---
g_opt.zero_grad()
generated_data = generator(noise, fake_labels)
g_loss = loss(discriminator(generated_data, fake_labels).view(bs), ones)
g_loss.backward(); g_opt.step()
```
- **Visual:** Vẽ hai đường loss G_loss và D_loss theo epoch trên cùng đồ thị — kỳ vọng chúng 'giằng co' quanh một mức, không đường nào sụp về 0.
- **Lưu ý:** Nhãn ở bước G là ones (không phải zeros) — G muốn D tin ảnh giả là thật. Nhớ zero_grad() cho ĐÚNG optimizer trước mỗi backward. lr thường nhỏ (0.0002); lr quá cao làm GAN phân kỳ/mode collapse.
- *Nguồn: CGANs.txt*

#### detach() để chặn gradient sang Generator (Stop-gradient khi train D)

- **Là gì:** Khi tính loss của D trên ảnh giả, gọi generated_data.detach() để tách ảnh giả khỏi đồ thị tính đạo hàm, nên gradient của bước D không lan ngược vào G.
- **Ý nghĩa & mục đích:** Ở bước D, ta chỉ muốn cập nhật D, không đụng vào G. detach() ngăn đạo hàm chảy vào tham số G, vừa đúng logic vừa tiết kiệm tính toán.
- **Code:**

```python
# BUOC D: detach de khong update G
out_fake = discriminator(generated_data.detach(), fake_labels).view(bs)
fake_loss = loss(out_fake, zeros)

# BUOC G: KHONG detach, vi can gradient chay ve G
out_fake = discriminator(generated_data, fake_labels).view(bs)
g_loss = loss(out_fake, ones)
```
- **Visual:** Sơ đồ đồ thị tính toán: mũi tên gradient bị 'cắt' tại chỗ detach ở bước D; ở bước G mũi tên chạy suốt từ loss qua D về tới trọng số G.
- **Lưu ý:** Bẫy kinh điển: quên detach ở bước D -> G bị cập nhật sai theo hướng của D. Ngược lại, ở bước G TUYỆT ĐỐI không detach, nếu không G không nhận gradient và không học gì.
- *Nguồn: CGANs.txt*

#### Nhãn thật/giả và BCELoss (Real/fake targets với Binary Cross-Entropy)

- **Là gì:** Tạo vector ones (nhãn thật=1) và zeros (nhãn giả=0), dùng nn.BCELoss so output Sigmoid của D với các nhãn này.
- **Ý nghĩa & mục đích:** GAN quy về bài phân loại nhị phân thật/giả, nên BCELoss là loss tự nhiên. ones/zeros là 'đáp án' cho D; ở bước G ta cố ý dùng ones cho ảnh giả để lừa D.
- **Code:**

```python
loss = nn.BCELoss()
ones  = torch.ones(batch_size).to(device)
zeros = torch.zeros(batch_size).to(device)
true_loss = loss(discriminator(true_data).view(batch_size), ones)
fake_loss = loss(discriminator(generated.detach()).view(batch_size), zeros)
```
- **Visual:** Bảng nhỏ 2 cột: (ảnh thật -> target 1), (ảnh giả, bước D -> target 0), (ảnh giả, bước G -> target 1). Giúp nhớ ai gán nhãn gì.
- **Lưu ý:** BCELoss cần input đã qua Sigmoid trong [0,1]; nếu D xuất logit thô sẽ sai/NaN. Phải .view(batch_size) cho output của D khớp shape với ones/zeros. Modern: bỏ Sigmoid trong D và dùng nn.BCEWithLogitsLoss để ổn định số học hơn.
- *Nguồn: CGANs.txt*

#### Vector nhiễu đầu vào (Noise / latent vector)

- **Là gì:** Vector ngẫu nhiên chuẩn (torch.randn) kích thước 100, làm hạt giống để Generator sinh ảnh.
- **Ý nghĩa & mục đích:** Là 'không gian tiềm ẩn' (latent space): mỗi nhiễu khác nhau cho một ảnh khác nhau, nhờ đó GAN sinh được đa dạng mẫu thay vì một ảnh cố định.
- **Code:**

```python
noise = torch.randn(batch_size, 100).to(device)
generated_data = generator(noise, fake_labels)
```
- **Visual:** Có thể nội suy giữa hai vector nhiễu z1, z2 và vẽ chuỗi ảnh sinh dọc đường nội suy để thấy latent space 'mượt' ra sao.
- **Lưu ý:** Dùng randn (phân phối chuẩn) chứ không rand (đều 0-1). Kích thước 100 phải khớp input_dim của Generator; sai chiều là lỗi matmul.
- *Nguồn: CGANs.txt*

#### LeakyReLU trong GAN (Leaky ReLU activation)

- **Là gì:** Biến thể ReLU cho phép độ dốc nhỏ (0.2) ở phần âm thay vì cắt về 0 hoàn toàn.
- **Ý nghĩa & mục đích:** Trong GAN, LeakyReLU giúp gradient vẫn chảy qua vùng âm, tránh 'neuron chết' — quan trọng cho ổn định huấn luyện GAN vốn đã mong manh.
- **Code:**

```python
nn.LeakyReLU(0.2)   # do doc 0.2 o phan am
```
- **Visual:** Vẽ hàm kích hoạt: đường gãy tại 0, bên phải dốc 1, bên trái dốc 0.2 (không phẳng như ReLU thường).
- **Lưu ý:** Đây là lựa chọn kinh nghiệm cho GAN (tutorial ghi rõ 'Note the use of Leaky RELUs'). Hệ số 0.2 là quy ước phổ biến; đừng nhầm với ReLU thường vốn phẳng ở phần âm.
- *Nguồn: CGANs.txt*

#### Sinh và vẽ lưới ảnh trong chế độ no_grad (Sampling & plotting a grid)

- **Là gì:** Sau khi train, lấy mẫu ảnh bằng cách đưa nhiễu (và nhãn nếu CGAN) qua generator trong torch.no_grad(), rồi reshape 28x28 và vẽ lưới có tiêu đề.
- **Ý nghĩa & mục đích:** Đây là cách 'nhìn' xem GAN học được gì — kiểm định định tính. no_grad() tắt việc dựng đồ thị đạo hàm khi chỉ suy diễn, tiết kiệm bộ nhớ và nhanh hơn.
- **Code:**

```python
num_imgs = 12
with torch.no_grad():
    noise = torch.randn(num_imgs, 100).to(device)
    fake_labels = torch.randint(0, 10, (num_imgs,)).to(device)
    generated = c_generator(noise, fake_labels).cpu().view(num_imgs, 28, 28)
    plot_images(generated, fake_labels.cpu().numpy())
```
- **Visual:** Lưới ncols ảnh, mỗi ô là một chữ số sinh ra; với CGAN đặt title = nhãn yêu cầu để đối chiếu ảnh có khớp nhãn không.
- **Lưu ý:** Phải .cpu() trước khi .numpy()/vẽ nếu tensor đang ở GPU. Với GAN không điều kiện thì gọi generator(noise) (không truyền labels) và đặt title = -1. plot_images kỳ vọng shape (n, 28, 28).
- *Nguồn: CGANs.txt*

#### Bất ổn định GAN & sụp chế độ (GAN instability / mode collapse)

- **Là gì:** Hiện tượng GAN khó hội tụ: nếu D quá mạnh, G mất gradient hữu ích; nếu G tìm được một vài mẫu lừa được D, nó lặp lại chúng và bỏ qua phần còn lại của phân phối (mode collapse).
- **Ý nghĩa & mục đích:** Hiểu điều này để không hoảng khi loss dao động, và để biết các 'nút chỉnh' ổn định: lr nhỏ (0.0002), Adam, Dropout trong D, lấy trung bình hai loss của D, LeakyReLU. Đây là kiến thức tái sử dụng cho mọi lần train GAN.
- **Code:**

```python
# cac lua chon giup on dinh:
d_opt = optim.Adam(discriminator.parameters(), lr=0.0002)
g_opt = optim.Adam(generator.parameters(), lr=0.0002)
d_loss = (real_loss + fake_loss) / 2   # can bang tin hieu cho D
# D co Dropout(0.3), ca hai dung LeakyReLU(0.2)
```
- **Visual:** Vẽ G_loss và D_loss theo epoch: lành mạnh là hai đường giằng co quanh một mức; dấu hiệu xấu là D_loss sụp về 0 (D thắng tuyệt đối) hoặc lưới ảnh sinh ra chỉ toàn một/vài chữ số giống nhau (mode collapse).
- **Lưu ý:** Đừng đọc loss GAN như loss phân lớp — loss giảm đều KHÔNG phải mục tiêu. Chẩn đoán chất lượng bằng cách nhìn ảnh sinh ra, không chỉ nhìn số loss. lr quá cao hoặc D áp đảo G là nguyên nhân phổ biến gây phân kỳ/collapse.
- *Nguồn: CGANs.txt*

#### Chuẩn hoá dữ liệu về [-1,1] cho GAN (Normalize để khớp Tanh)

- **Là gì:** Dùng transforms.Normalize(mean=[0.5], std=[0.5]) để đưa ảnh MNIST từ [0,1] về [-1,1].
- **Ý nghĩa & mục đích:** Phải khớp với output Tanh của Generator (cũng [-1,1]). Nếu thang giá trị của ảnh thật và ảnh giả khác nhau, D phân biệt được ngay lập tức và G không học nổi.
- **Code:**

```python
transform_list = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])])  # [0,1] -> [-1,1]
```
- **Visual:** Vẽ histogram giá trị pixel trước và sau normalize: trước nằm [0,1], sau trải ra [-1,1] và tâm về 0.
- **Lưu ý:** Bẫy ngầm: quên bước này (hoặc dùng thang khác) làm G không hội tụ dù kiến trúc đúng. Normalize dùng công thức (x-mean)/std nên (x-0.5)/0.5 mới ra [-1,1].
- *Nguồn: CGANs.txt*

#### Chuẩn hoá ảnh cho metric learning (Zero-mean & chia độ lệch chuẩn chung)

- **Là gì:** Trừ ảnh trung bình khỏi từng ảnh (zero mean theo pixel), rồi chia TẤT CẢ pixel cho một độ lệch chuẩn chung của toàn bộ pixel (std_train).
- **Ý nghĩa & mục đích:** Đưa giá trị về tầm nhỏ, hợp lý (~[-2,4]) giúp học nhanh giai đoạn đầu và tránh bão hoà neuron Tanh. Chia bằng std CHUNG (không phải std từng pixel) để tránh thổi phồng những pixel gần như luôn bằng 0.
- **Code:**

```python
x_train = mnist_trainset.data.type(torch.DoubleTensor)
x_train = x_train - torch.mean(x_train, dim=0)      # zero-mean theo pixel
std_train = torch.std(x_train.reshape(-1))           # 1 std chung cho tat ca pixel
x_train = x_train / std_train
# test: chia bang std cua TRAIN
x_test = mnist_testset.data.type(torch.DoubleTensor)
x_test = (x_test - torch.mean(x_test, dim=0)) / std_train
```
- **Visual:** plt.imshow(mean_image) để thấy pixel nào quan trọng; plt.hist(x_train.reshape(-1)) để xác nhận giá trị đã gọn về khoảng [-2,4].
- **Lưu ý:** Chia theo std TỪNG pixel là sai: pixel hiếm khi khác 0 sẽ bị khuếch đại khổng lồ. Về std, test phải dùng std_train (thống kê của train) để không lệch thang. Lưu ý: bản gốc trừ test bằng CHÍNH mean của test (chỉ dùng chung std_train) — chặt chẽ hơn thì nên trừ luôn bằng mean của train để tránh rò rỉ.
- *Nguồn: MNIST_Siamese.txt*


---

### 8.9 Transfer Learning & Vệ sinh huấn luyện

#### Transfer Learning (Học chuyển giao)

- **Là gì:** Lấy một mạng đã được huấn luyện sẵn trên tập dữ liệu rất lớn (vd ImageNet: 1.2 triệu ảnh, 1000 lớp) rồi tái sử dụng cho bài toán mới của mình, thay vì huấn luyện từ đầu với trọng số ngẫu nhiên.
- **Ý nghĩa & mục đích:** Rất ít người huấn luyện cả một ConvNet từ số 0 vì hiếm khi có đủ dữ liệu. Ở đây tập ants/bees chỉ ~120 ảnh train mỗi lớp — nếu train from scratch sẽ không tổng quát nổi. Transfer learning cho phép tận dụng đặc trưng đã học sẵn để vẫn đạt kết quả tốt trên dữ liệu nhỏ.
- **Code:**

```python
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # thay lớp cuối cho số lớp mới
```
- **Visual:** Vẽ sơ đồ khối: [Backbone ImageNet đã học] -> [Lớp FC mới random] -> [2 lớp: ant/bee]. Hoặc so sánh 2 đường accuracy (from-scratch thấp/dao động vs transfer cao/ổn định) trên cùng tập nhỏ.
- **Lưu ý:** Chỉ hiệu quả khi domain nguồn và đích đủ giống nhau (ảnh tự nhiên -> ảnh tự nhiên). Tutorial gốc dùng models.resnet18(pretrained=True) — tham số pretrained đã bị torchvision loại bỏ (deprecated), thay bằng weights=... (vd weights=models.ResNet18_Weights.IMAGENET1K_V1 hoặc weights='DEFAULT'); pretrained=True là kiểu cũ chỉ còn để tương thích.
- *Nguồn: transfer_learning_tutorial.txt*

#### Finetuning toàn mạng (Finetuning the convnet)

- **Là gì:** Kịch bản transfer learning thứ nhất: khởi tạo mạng bằng trọng số pretrained rồi tiếp tục huấn luyện TẤT CẢ các tham số như bình thường (không đóng băng gì cả).
- **Ý nghĩa & mục đích:** Dùng khi muốn cả backbone tự điều chỉnh theo dữ liệu mới; pretrained chỉ đóng vai trò điểm khởi đầu tốt thay cho random init. Cho độ chính xác cao nhất nhưng train lâu hơn vì phải tính gradient cho mọi tham số.
- **Code:**

```python
model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
# tất cả tham số đều được tối ưu
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
```
- **Visual:** Sơ đồ mạng với TẤT CẢ khối tô màu 'trainable' (mũi tên gradient chạy ngược qua toàn bộ). Đối lập với ảnh feature-extractor chỉ tô lớp cuối.
- **Lưu ý:** Truyền model_ft.parameters() (toàn bộ) vào optimizer — khác với feature extractor chỉ truyền model.fc.parameters(). Lâu hơn đáng kể trên CPU (15-25 phút vs nửa thời gian). Lưu ý pretrained=True trong tutorial là API cũ, dùng weights=... trên torchvision hiện đại.
- *Nguồn: transfer_learning_tutorial.txt*

#### ConvNet như bộ trích đặc trưng cố định (ConvNet as fixed feature extractor)

- **Là gì:** Kịch bản transfer learning thứ hai: đóng băng (freeze) toàn bộ mạng trừ lớp fully-connected cuối; lớp cuối bị thay bằng lớp mới random và CHỈ lớp này được huấn luyện.
- **Ý nghĩa & mục đích:** Dùng khi dữ liệu rất nhỏ hoặc muốn train nhanh: coi backbone như một máy trích đặc trưng bất biến, chỉ học lại 'bộ phân loại' ở đầu ra. Nhanh hơn nhiều vì gradient không cần tính cho phần lớn mạng (nhưng forward vẫn phải chạy qua toàn bộ).
- **Code:**

```python
model_conv = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model_conv.parameters():
    param.requires_grad = False
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)  # lớp mới có requires_grad=True mặc định
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
```
- **Visual:** Sơ đồ mạng: các khối backbone tô xám ('frozen', không có mũi tên gradient), chỉ lớp FC cuối tô màu sáng ('trainable').
- **Lưu ý:** Phải truyền model_conv.fc.parameters() (chỉ lớp cuối) vào optimizer, không phải toàn bộ. Nếu quên freeze mà vẫn chỉ đưa fc vào optimizer thì backbone không cập nhật nhưng vẫn tốn tính gradient thừa. Freeze phải làm TRƯỚC khi thay lớp fc. Dùng weights=... thay cho pretrained=True (đã deprecated).
- *Nguồn: transfer_learning_tutorial.txt*

#### Đóng băng tham số (Freezing / requires_grad = False)

- **Là gì:** Đặt param.requires_grad = False cho một tham số để autograd KHÔNG tính gradient cho nó trong backward(), khiến optimizer không cập nhật tham số đó.
- **Ý nghĩa & mục đích:** Cơ chế kỹ thuật để 'khóa' phần mạng ta không muốn thay đổi (backbone pretrained). Loại các nhánh con khỏi đồ thị backward, tiết kiệm tính toán và giữ nguyên đặc trưng đã học.
- **Code:**

```python
for param in model_conv.parameters():
    param.requires_grad = False
# module mới tạo (nn.Linear) có requires_grad=True theo mặc định
```
- **Visual:** Bảng liệt kê param.name + requires_grad (True/False) để kiểm tra layer nào còn học. Hoặc histogram: đếm số tham số trainable vs frozen.
- **Lưu ý:** Tham số của module mới tạo (như nn.Linear thay fc) mặc định requires_grad=True — không cần bật lại. Đóng băng chỉ chặn cập nhật, forward vẫn chạy qua các lớp đó nên vẫn tốn thời gian forward.
- *Nguồn: transfer_learning_tutorial.txt*

#### Thay lớp phân loại cuối (Replacing the final FC layer)

- **Là gì:** Lấy số chiều đầu vào của lớp fc gốc (in_features) rồi gán model.fc = nn.Linear(num_ftrs, num_classes) để khớp số lớp của bài toán mới.
- **Ý nghĩa & mục đích:** Mạng pretrained có đầu ra 1000 lớp (ImageNet); bài toán mới có số lớp khác (ở đây 2). Phải thay lớp cuối để đúng kích thước đầu ra; lớp mới khởi tạo random và được học lại.
- **Code:**

```python
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
# tổng quát hơn: nn.Linear(num_ftrs, len(class_names))
```
- **Visual:** Sơ đồ: hộp '512 -> 1000' bị gạch bỏ, thay bằng hộp '512 -> 2'. Ghi rõ in_features lấy từ lớp cũ.
- **Lưu ý:** Nên dùng len(class_names) thay vì hard-code số 2 để không sai khi đổi tập dữ liệu (tutorial cũng ghi chú điều này). Đọc in_features từ lớp gốc TRƯỚC khi ghi đè, nếu không sẽ mất thông tin kích thước.
- *Nguồn: transfer_learning_tutorial.txt*

#### Vòng lặp train/val theo epoch (Train/validation phase loop)

- **Là gì:** Mỗi epoch chạy hai pha: 'train' (cập nhật trọng số) và 'val' (chỉ đánh giá). Chuyển chế độ mô hình và bật/tắt tính gradient tùy pha.
- **Ý nghĩa & mục đích:** Tách rõ giai đoạn học và giai đoạn đo lường trên cùng vòng lặp, để theo dõi loss/accuracy trên cả hai và phát hiện overfitting. Đây là khung sườn train chuẩn tái sử dụng được.
- **Code:**

```python
for phase in ['train', 'val']:
    if phase == 'train':
        model.train()
    else:
        model.eval()
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            if phase == 'train':
                loss.backward()
                optimizer.step()
    if phase == 'train':
        scheduler.step()
```
- **Visual:** Hai đường loss (train vs val) theo epoch trên cùng biểu đồ — train giảm đều, val chững/tăng là dấu hiệu overfit.
- **Lưu ý:** Chỉ gọi loss.backward()/optimizer.step() ở pha train. Quên chuyển model.eval() ở pha val sẽ làm BatchNorm/Dropout hoạt động sai, méo kết quả đánh giá. scheduler.step() gọi MỘT LẦN sau khi hết pha train, không phải trong vòng lặp batch.
- *Nguồn: transfer_learning_tutorial.txt*

#### Chế độ train vs eval của mô hình (model.train() / model.eval())

- **Là gì:** Hai công tắc chế độ của nn.Module: train() bật hành vi huấn luyện, eval() bật hành vi suy luận cho các lớp phụ thuộc chế độ (Dropout, BatchNorm).
- **Ý nghĩa & mục đích:** Dropout và BatchNorm cư xử khác nhau khi train và khi đánh giá (Dropout tắt ở eval; BatchNorm dùng thống kê chạy thay vì thống kê batch). Đặt đúng chế độ để kết quả val/inference chính xác và tái lập được.
- **Code:**

```python
model.train()  # trước khi huấn luyện
model.eval()   # trước khi đánh giá / dự đoán
# lưu và khôi phục chế độ:
was_training = model.training
model.eval()
# ...
model.train(mode=was_training)
```
- **Visual:** Bảng đối chiếu: Dropout (on/off), BatchNorm (batch stats / running stats) theo hai chế độ train/eval.
- **Lưu ý:** eval() KHÔNG tắt việc tính gradient — vẫn cần torch.no_grad()/set_grad_enabled để khỏi tốn bộ nhớ. Quên gọi eval() khi đánh giá là lỗi phổ biến làm accuracy dao động vô lý.
- *Nguồn: transfer_learning_tutorial.txt*

#### Bật/tắt tính gradient (torch.set_grad_enabled / torch.no_grad)

- **Là gì:** Context manager bật hoặc tắt việc autograd theo dõi lịch sử để tính gradient. set_grad_enabled(cond) bật theo điều kiện; no_grad() luôn tắt.
- **Ý nghĩa & mục đích:** Ở pha val/inference không cần gradient nên tắt đi để tiết kiệm bộ nhớ và tăng tốc. set_grad_enabled(phase=='train') gộp gọn logic cho cả hai pha trong một khối with.
- **Code:**

```python
with torch.set_grad_enabled(phase == 'train'):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
# khi chỉ dự đoán:
with torch.no_grad():
    outputs = model(inputs)
```
- **Visual:** Sơ đồ đồ thị tính toán: khi bật, các nút lưu lại để backward; khi tắt, đồ thị không được dựng (mũi tên ngược biến mất).
- **Lưu ý:** Đây là chuyện KHÁC với model.eval() — cần cả hai. Chỉ tắt grad thôi thì Dropout/BatchNorm vẫn ở chế độ train nếu chưa gọi eval(). Không còn dùng torch.autograd.Variable để bật/tắt grad — Variable đã bị gộp vào Tensor và loại bỏ từ lâu.
- *Nguồn: transfer_learning_tutorial.txt*

#### Xóa gradient trước mỗi bước (optimizer.zero_grad())

- **Là gì:** Đặt lại gradient của mọi tham số về 0 trước khi gọi backward() cho batch mới.
- **Ý nghĩa & mục đích:** PyTorch cộng dồn (accumulate) gradient qua các lần backward. Nếu không xóa, gradient batch trước cộng vào batch sau làm bước cập nhật sai. Gọi ở đầu mỗi vòng lặp batch để mỗi bước độc lập.
- **Code:**

```python
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```
- **Visual:** Timeline 3 bước lặp: thanh gradient reset về 0 ở đầu mỗi bước; nếu bỏ zero_grad thì thanh cứ chồng cao dần (minh họa cộng dồn sai).
- **Lưu ý:** Quên zero_grad() là bug kinh điển: loss trông vẫn giảm nhưng bước cập nhật bị nhiễu do gradient tích lũy. Thứ tự chuẩn: zero_grad -> forward -> backward -> step.
- *Nguồn: transfer_learning_tutorial.txt*

#### Lập lịch learning rate (LR scheduler — StepLR)

- **Là gì:** Đối tượng từ torch.optim.lr_scheduler tự động giảm learning rate theo lịch. StepLR nhân LR với gamma sau mỗi step_size epoch.
- **Ý nghĩa & mục đích:** Giảm dần LR giúp mạng học nhanh lúc đầu rồi tinh chỉnh mượt về cuối, tránh dao động quanh cực tiểu. Ở đây LR giảm 10 lần (gamma=0.1) mỗi 7 epoch.
- **Code:**

```python
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# trong vòng train, gọi SAU khi hết pha train mỗi epoch:
if phase == 'train':
    scheduler.step()
```
- **Visual:** Đường LR theo epoch dạng bậc thang: phẳng 7 epoch rồi tụt xuống 1/10, lặp lại. Có thể ghi kèm đường loss để thấy loss mượt hơn sau mỗi lần tụt.
- **Lưu ý:** Gọi scheduler.step() MỘT LẦN mỗi epoch (sau pha train), không phải mỗi batch. Trong PyTorch mới, thứ tự đúng là optimizer.step() trước rồi scheduler.step(); gọi sai thứ tự sẽ bị cảnh báo bỏ lỡ LR đầu.
- *Nguồn: transfer_learning_tutorial.txt*

#### Optimizer SGD với momentum (optim.SGD)

- **Là gì:** Bộ tối ưu Stochastic Gradient Descent kèm momentum, cập nhật tham số theo gradient và một phần bước trước đó.
- **Ý nghĩa & mục đích:** Momentum giúp vượt qua vùng phẳng và giảm dao động, hội tụ nhanh và ổn định hơn SGD trơn. Tham số quan trọng: lr (tốc độ học) và momentum. Nhận danh sách tham số cần tối ưu ở đối số đầu.
- **Code:**

```python
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# feature extractor: chỉ truyền tham số lớp cuối
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
```
- **Visual:** So sánh quỹ đạo hội tụ trên mặt loss: SGD thường zíc-zắc, SGD+momentum mượt và thẳng hơn về đáy.
- **Lưu ý:** Danh sách tham số truyền vào quyết định phần nào của mạng được cập nhật — .parameters() (toàn bộ) cho finetune, .fc.parameters() cho feature extractor. LR quá cao (vd 0.1) dễ gây phân kỳ (loss tăng vọt).
- *Nguồn: transfer_learning_tutorial.txt*

#### Hàm mất mát CrossEntropyLoss (nn.CrossEntropyLoss)

- **Là gì:** Hàm loss cho phân loại nhiều lớp, nhận logits thô (chưa qua softmax) và nhãn lớp dạng chỉ số nguyên.
- **Ý nghĩa & mục đích:** Chuẩn cho bài toán phân loại đa lớp. Gộp sẵn LogSoftmax + NLLLoss nên đầu ra model chỉ cần là logits, không cần tự softmax.
- **Code:**

```python
criterion = nn.CrossEntropyLoss()
outputs = model(inputs)          # logits thô, shape [N, num_classes]
loss = criterion(outputs, labels)  # labels là chỉ số lớp, không one-hot
```
- **Visual:** Minh họa: vector logits -> softmax thành xác suất -> lấy -log của xác suất lớp đúng. Vẽ đường loss giảm theo epoch.
- **Lưu ý:** KHÔNG thêm softmax vào đầu ra model rồi mới đưa vào CrossEntropyLoss — sẽ tính softmax hai lần, sai. Nhãn là chỉ số nguyên (LongTensor), không phải one-hot.
- *Nguồn: transfer_learning_tutorial.txt*

#### Lấy dự đoán từ logits (torch.max theo chiều lớp)

- **Là gì:** torch.max(outputs, 1) trả về (giá trị lớn nhất, chỉ số lớp lớn nhất) theo chiều 1; chỉ số chính là lớp dự đoán.
- **Ý nghĩa & mục đích:** Chuyển logits thành nhãn dự đoán để tính accuracy. Chiều 1 là chiều các lớp; ta chỉ cần argmax nên bỏ giá trị, giữ chỉ số preds.
- **Code:**

```python
outputs = model(inputs)
_, preds = torch.max(outputs, 1)
running_corrects += torch.sum(preds == labels.data)
```
- **Visual:** Bảng nhỏ: hàng logits [ant=1.2, bee=3.4] -> mũi tên argmax -> preds=bee. Có thể vẽ confusion matrix từ preds vs labels.
- **Lưu ý:** Nhớ chọn đúng chiều (dim=1 cho batch [N, C]); dùng dim=0 sẽ lấy max sai trục. Dấu _ bỏ giá trị max vì chỉ cần chỉ số. Có thể thay bằng torch.argmax(outputs, 1) nếu chỉ cần chỉ số.
- *Nguồn: transfer_learning_tutorial.txt*

#### Lưu và khôi phục mô hình tốt nhất (best model via state_dict + deepcopy)

- **Là gì:** Trong lúc train, mỗi khi accuracy trên val vượt kỷ lục thì deepcopy state_dict lại; kết thúc train thì load lại bộ trọng số tốt nhất đó.
- **Ý nghĩa & mục đích:** Mô hình ở epoch cuối chưa chắc tốt nhất (có thể đã overfit). Giữ lại snapshot theo val accuracy đảm bảo trả về phiên bản tổng quát tốt nhất — một dạng early-stopping thủ công.
- **Code:**

```python
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
# ... trong pha val:
if phase == 'val' and epoch_acc > best_acc:
    best_acc = epoch_acc
    best_model_wts = copy.deepcopy(model.state_dict())
# cuối cùng:
model.load_state_dict(best_model_wts)
```
- **Visual:** Đường val accuracy theo epoch với một dấu chấm đánh dấu đỉnh cao nhất (điểm được lưu), khác với điểm cuối cùng.
- **Lưu ý:** Phải deepcopy state_dict, không gán trực tiếp — state_dict trỏ tới cùng tensor, không copy thì 'bản lưu' bị ghi đè khi train tiếp. Chọn best theo val accuracy, không theo train. Khi lưu ra đĩa nên lưu state_dict (torch.save(model.state_dict(), path)) chứ đừng pickle cả object model — cách lưu cả model dễ vỡ khi đổi mã nguồn.
- *Nguồn: transfer_learning_tutorial.txt*

#### Tăng cường dữ liệu ảnh (Data augmentation — train transforms)

- **Là gì:** Biến đổi ngẫu nhiên ảnh lúc train (cắt-thu-phóng ngẫu nhiên, lật ngang) để tạo biến thể; lúc val chỉ resize + crop giữa cố định.
- **Ý nghĩa & mục đích:** Tăng đa dạng dữ liệu train giả tạo, giúp mô hình tổng quát tốt hơn và giảm overfit — đặc biệt quan trọng khi tập nhỏ. Val KHÔNG augment ngẫu nhiên để đánh giá ổn định, tái lập.
- **Code:**

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}
```
- **Visual:** Lưới ảnh: một ảnh gốc -> nhiều biến thể (crop/lật khác nhau). Dùng imshow trên make_grid của một batch train để thấy augmentation.
- **Lưu ý:** Chỉ augment ngẫu nhiên ở pha train, KHÔNG ở val/test (nếu không kết quả sẽ nhiễu, không so sánh được). Kích thước crop phải khớp đầu vào mạng (224 cho resnet18).
- *Nguồn: transfer_learning_tutorial.txt*

#### Chuẩn hóa theo mean/std của ImageNet (transforms.Normalize)

- **Là gì:** Chuẩn hóa mỗi kênh màu bằng mean [0.485,0.456,0.406] và std [0.229,0.224,0.225] — đúng thống kê của ImageNet.
- **Ý nghĩa & mục đích:** Mạng pretrained trên ImageNet kỳ vọng đầu vào được chuẩn hóa đúng như lúc nó được train. Dùng sai mean/std sẽ làm phân phối đầu vào lệch, đặc trưng pretrained hoạt động kém.
- **Code:**

```python
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# đảo ngược để hiển thị:
inp = std * inp + mean
inp = np.clip(inp, 0, 1)
```
- **Visual:** Histogram giá trị pixel trước và sau normalize (dịch về quanh 0). Khi hiển thị phải 'un-normalize' rồi clip [0,1] mới ra ảnh đúng màu.
- **Lưu ý:** Cùng bộ mean/std phải dùng cho cả train và val. Muốn xem ảnh phải đảo ngược normalize (std*x+mean) rồi clip, nếu không ảnh sẽ méo màu/âm giá trị.
- *Nguồn: transfer_learning_tutorial.txt*

#### Nạp ảnh theo thư mục lớp (ImageFolder + DataLoader)

- **Là gì:** datasets.ImageFolder đọc ảnh từ cấu trúc thư mục (mỗi lớp một folder con) và tự gán nhãn; DataLoader chia batch, xáo trộn, nạp song song.
- **Ý nghĩa & mục đích:** Cách chuẩn, gọn để dựng pipeline dữ liệu ảnh: tên thư mục con thành class_names, dataset_sizes để tính loss/acc trung bình đúng. shuffle=True cho train giúp học ổn định.
- **Code:**

```python
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
               shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
```
- **Visual:** Sơ đồ cây thư mục: data_dir/train/ants/*, data_dir/train/bees/* -> nhãn 0/1. Vẽ một batch bằng make_grid + imshow.
- **Lưu ý:** num_workers>0 trên Windows có thể lỗi nếu không đặt trong if __name__=='__main__'. class_names lấy từ thứ tự alphabet của thư mục — nhớ ánh xạ đúng khi đọc dự đoán.
- *Nguồn: transfer_learning_tutorial.txt*

#### Đưa dữ liệu và mô hình lên đúng thiết bị (device / .to(device))

- **Là gì:** Chọn cuda nếu có, ngược lại cpu; rồi chuyển cả model lẫn tensor đầu vào/nhãn sang thiết bị đó bằng .to(device).
- **Ý nghĩa & mục đích:** Model và dữ liệu phải nằm CÙNG thiết bị mới tính được. Viết device một lần rồi .to(device) khắp nơi giúp code chạy được cả trên GPU (nhanh) lẫn CPU mà không sửa gì.
- **Code:**

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)
```
- **Visual:** Sơ đồ hai hộp CPU/GPU với mũi tên .to(device) chuyển tensor giữa chúng; nhắc model + data phải cùng phía.
- **Lưu ý:** Quên .to(device) cho inputs/labels khi model ở GPU sẽ báo lỗi 'expected all tensors on same device'. Khi hiển thị ảnh phải .cpu() trước (inputs.cpu().data[j]).
- *Nguồn: transfer_learning_tutorial.txt*

#### Tính loss và accuracy trung bình theo epoch (running_loss / running_corrects)

- **Là gì:** Cộng dồn loss có trọng số theo kích thước batch và số dự đoán đúng qua các batch, rồi chia cho tổng số mẫu để ra loss/acc trung bình của epoch.
- **Ý nghĩa & mục đích:** Cho con số loss/accuracy đại diện cả tập (không bị lệch do batch cuối nhỏ). Nhân loss.item() với inputs.size(0) để cộng đúng theo số mẫu thực trong batch.
- **Code:**

```python
running_loss += loss.item() * inputs.size(0)
running_corrects += torch.sum(preds == labels.data)
epoch_loss = running_loss / dataset_sizes[phase]
epoch_acc = running_corrects.double() / dataset_sizes[phase]
```
- **Visual:** Hai đường epoch_loss và epoch_acc theo epoch cho cả train và val trên một dashboard nhỏ.
- **Lưu ý:** Dùng loss.item() (số Python) chứ không giữ tensor loss để tránh giữ đồ thị tính toán gây rò rỉ bộ nhớ. Chia cho dataset_sizes[phase], không phải số batch. running_corrects cần .double() trước khi chia để ra tỉ lệ thực.
- *Nguồn: transfer_learning_tutorial.txt*

#### Trực quan hóa dự đoán của mô hình (visualize_model)

- **Là gì:** Hàm tiện ích chạy mô hình ở chế độ eval, lấy vài ảnh val, hiển thị ảnh kèm nhãn dự đoán, rồi khôi phục lại chế độ train ban đầu.
- **Ý nghĩa & mục đích:** Kiểm tra định tính xem mô hình đoán đúng/sai thế nào trên ảnh thật, thay vì chỉ nhìn con số accuracy. Lưu/khôi phục was_training để không làm hỏng trạng thái mô hình sau khi gọi hàm.
- **Code:**

```python
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            # ... imshow + set_title('predicted: '+class_names[preds[j]])
    model.train(mode=was_training)
```
- **Visual:** Lưới ô ảnh val, mỗi ô là một ảnh + tiêu đề 'predicted: ant/bee'. Đúng nghĩa là 'visual' của khái niệm này.
- **Lưu ý:** Nhớ khôi phục model.train(mode=was_training) trước khi return (kể cả nhánh return sớm) để không để mô hình kẹt ở eval. Bọc trong torch.no_grad() để khỏi tốn bộ nhớ.
- *Nguồn: transfer_learning_tutorial.txt*

#### Lấy một batch từ DataLoader (next(iter(dataloader)))

- **Là gì:** Tạo iterator từ DataLoader rồi lấy phần tử đầu tiên để có một batch (inputs, labels) — dùng để xem thử dữ liệu hoặc debug shape.
- **Ý nghĩa & mục đích:** Trước khi train, ta thường muốn nhìn một batch mẫu (kiểm tra augmentation, shape, nhãn). next(iter(...)) là cách nhanh gọn lấy đúng một batch mà không phải viết vòng lặp.
- **Code:**

```python
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])
```
- **Visual:** Một lưới ảnh (make_grid) của batch đầu tiên, tiêu đề là danh sách nhãn ant/bee tương ứng.
- **Lưu ý:** API cũ dataiter.next() đã bị bỏ trong Python hiện đại — dùng next(iter(dataloader)) hoặc next(dataiter). Mỗi lần next(iter(...)) tạo iterator mới nên luôn trả batch đầu; muốn duyệt tiếp phải giữ lại iterator.
- *Nguồn: transfer_learning_tutorial.txt*

#### Hiển thị tensor ảnh đã chuẩn hóa (imshow helper)

- **Là gì:** Hàm chuyển tensor ảnh [C,H,W] về [H,W,C], đảo ngược normalize (std*x+mean), clip về [0,1] rồi vẽ bằng plt.imshow.
- **Ý nghĩa & mục đích:** Tensor sau ToTensor + Normalize không hiển thị đúng màu (giá trị âm, sai thứ tự trục). Hàm imshow tái sử dụng để 'giải mã' tensor về ảnh xem được — cần cho mọi bước debug dữ liệu ảnh.
- **Code:**

```python
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
```
- **Visual:** So sánh cạnh nhau: tensor thô (màu méo/âm) vs sau un-normalize + clip (ảnh đúng màu).
- **Lưu ý:** Tensor phải ở CPU trước khi .numpy() (gọi .cpu() nếu đang trên GPU). Thứ tự trục PyTorch là [C,H,W], phải transpose sang [H,W,C] cho matplotlib. Quên clip [0,1] sẽ bị matplotlib cảnh báo giá trị ngoài khoảng.
- *Nguồn: transfer_learning_tutorial.txt*


---

### 8.10 Regularisation & Generalisation (lõi luận văn)

#### Điều chuẩn (Regularisation)

- **Là gì:** Thêm một thành phần phạt vào hàm mất mát để 'kéo' mô hình về những nghiệm đơn giản/ổn định hơn, thay vì để nó tự do chọn bất kỳ nghiệm nào khớp dữ liệu.
- **Ý nghĩa & mục đích:** Dùng khi có nhiều nghiệm 'tốt như nhau' trên tập train nhưng một số nghiệm có trọng số rất lớn/rất nhỏ (gây bất ổn) hoặc khớp nhiễu (overfit). Regularisation giúp chọn nghiệm 'lành mạnh' hơn, ổn định khi học và tổng quát hoá tốt hơn trên dữ liệu mới. Ba dạng dùng trong khoá: L1, L2, dropout.
- **Code:**

```python
# L2 penalty added to the squared loss (Week1):
# J = 1/2 * ( (yhat - y)**2 + lam * (w1**2 + w2**2) )
# the lam*(w1**2 + w2**2) term punishes large values of w1 or w2
```
- **Visual:** Vẽ bề mặt mất mát: không regularise thì có cả một 'thung lũng' dài (nhiều nghiệm bằng nhau, w1*w2=1) nên nghiệm trôi tự do; thêm L2 tạo một 'giếng' quanh gốc kéo nghiệm về vùng trọng số nhỏ. Trong Week1 dùng plt.imshow(1/J_grid.transpose(), origin='lower') để nhìn vùng nghiệm.
- **Lưu ý:** Regularise quá mạnh sẽ ép mọi trọng số về 0 -> mô hình mất khả năng khớp (underfit). Phải cân bằng.
- *Nguồn: Week1Exercise2_SimpleDNN*

#### Phạt L2 / Suy giảm trọng số thủ công (L2 penalty, λ(w₁²+w₂²))

- **Là gì:** Thành phần phạt tỉ lệ với bình phương trọng số, cộng vào loss; đạo hàm của nó thêm một số hạng +λ·w vào gradient, kéo mỗi trọng số về gần 0 sau mỗi bước.
- **Ý nghĩa & mục đích:** Dùng để chống bất ổn khi một trọng số phình to (trong mô hình w1*w2*x, một weight lớn làm gradient của weight kia bùng nổ). L2 phạt cả w1 lẫn w2 nên không cho weight nào phình vô hạn.
- **Code:**

```python
regulariser = 0.01        # try larger and smaller values
learning_rate = 0.019
w1 = 0.7; w2 = 1.0/0.7
for x, y in zip(xx, yy):
    yhat = w1 * w2 * x
    grad = (yhat - y) * x
    w1 = w1 - learning_rate * grad * w2 - regulariser * w1
    w2 = w2 - learning_rate * grad * w1 - regulariser * w2
# dJ/dw1 = (yhat - y) * w2 * x + lam * w1
```
- **Visual:** Vẽ quỹ đạo (ww1, ww2) trên mặt phẳng: không regularise thì quỹ đạo trượt dọc hyperbol w1*w2=1 rồi bắn ra vô cực; có L2 thì quỹ đạo hội tụ về vùng w1≈w2≈1. plt.plot(ww1[:limit], ww2[:limit]).
- **Lưu ý:** regulariser quá lớn: yhat bị kéo lệch, dự đoán tệ đi (xem plt yy vs yhats). Quá nhỏ thì không ngăn được bùng nổ; phải dò bằng thử-sai. Lưu ý bản source cập nhật w2 bằng w1 ĐÃ mới (không phải song song); đây là cách viết gốc của bài, giữ nguyên cho trung thực.
- *Nguồn: Week1Exercise2_SimpleDNN*

#### Weight decay trong optimizer (weight_decay, L2 tự động)

- **Là gì:** Cách khai báo L2 cho cả mạng chỉ bằng một tham số của optimizer PyTorch; optimizer tự cộng phần phạt λ·w vào gradient của mọi trọng số, không cần viết tay.
- **Ý nghĩa & mục đích:** Dùng khi huấn luyện mạng thật (nhiều tham số) để bật/tắt và điều chỉnh L2 dễ dàng. Đây là công cụ chính cho 'Regularisation experiment' của đồ án: train nhiều lần với các mức weight_decay khác nhau rồi vẽ val_loss theo mức đó.
- **Code:**

```python
def define_and_train(NN_class, n_epochs, training_set, test_set,
                     batch_size=32, weight_decay=0.0):
    ...
    thenet = NN_class()
    optimizer1 = optim.Adam(thenet.parameters(), weight_decay=weight_decay)
    ...

# usage:
history1, net1 = define_and_train(NN_one_hidden, 50, training_set,
                                  test_set, batch_size=32, weight_decay=0.0001)
```
- **Visual:** Vẽ đường 'val_loss (hoặc error-rate) theo mức weight_decay' — trục x là lượng regularisation, trục y là hiệu năng validation; thường có hình chữ U: quá ít -> overfit, quá nhiều -> underfit, đáy là mức tối ưu.
- **Lưu ý:** Mức hợp lý cho L1/L2 khoảng 0.00001 đến 0.001; lớn hơn sẽ ép hết trọng số về 0. weight_decay=0.0 nghĩa là TẮT regularisation (dùng để cố tình overfit ở bước 1 của đồ án). Trong Adam, weight_decay là L2 kiểu cộng vào gradient; nếu muốn 'decoupled weight decay' đúng chuẩn thì dùng optim.AdamW.
- *Nguồn: Week3Exercise2_Percolation_CNN / CourseWorkProject*

#### Phạt L1 (L1 regularisation)

- **Là gì:** Dạng regularisation phạt theo giá trị tuyệt đối của trọng số (|w|) thay vì bình phương; xu hướng đẩy nhiều trọng số về đúng 0 (tạo mô hình 'thưa').
- **Ý nghĩa & mục đích:** Dùng như một lựa chọn thay thế L2 khi thử nghiệm loại regularisation nào giúp giảm overfit. Khoá học yêu cầu thử l1, l2 và dropout RIÊNG từng cái để so sánh.
- **Code:**

```python
# PyTorch's optimizer only builds in L2 (weight_decay); L1 must be added by hand:
l1_lambda = 0.0001  # reasonable range 0.00001 - 0.001
for images, labels in trainloader:
    optimizer1.zero_grad()
    outputs = thenet(images)
    l1_norm = sum(p.abs().sum() for p in thenet.parameters())
    loss = loss_function(outputs, labels) + l1_lambda * l1_norm
    loss.backward()
    optimizer1.step()
# Try l1, l2 and dropout separately, to compare.
```
- **Visual:** So sánh histogram trọng số sau khi train với L1 vs L2: L1 cho nhiều trọng số đúng bằng 0 (đỉnh nhọn tại 0), L2 cho trọng số nhỏ nhưng ít khi bằng 0.
- **Lưu ý:** Quá lớn ép hết weight về 0. PyTorch optimizer chỉ có weight_decay (L2) sẵn; L1 phải TỰ cộng lam*sum(|w|) vào loss như trên (không có tham số l1 trong optimizer). Đừng đưa cả bias vào phạt nếu không cần.
- *Nguồn: Week3Exercise2_Percolation_CNN*

#### Dropout

- **Là gì:** Kỹ thuật regularisation ngẫu nhiên 'tắt' một phần neuron trong lúc train, buộc mạng không phụ thuộc quá mức vào vài neuron cụ thể.
- **Ý nghĩa & mục đích:** Dùng để giảm overfit ở mạng nhiều tham số; là một trong ba loại regularisation được yêu cầu thử (l1, l2, dropout) và là lựa chọn hợp lệ cho 'Regularisation experiment' trong đồ án.
- **Code:**

```python
# add nn.Dropout(p) into the Sequential, between layers:
self.layers = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64, 100), nn.ReLU(),
    nn.Dropout(0.3),          # drops 30% of activations during training
    nn.Linear(100, 1), nn.Sigmoid())
# Does adding regularisation - l1, l2, or dropout - help? Try each separately.
```
- **Visual:** Vẽ 2 đường val_loss theo epoch: có dropout vs không; kỳ vọng đường có dropout tách khỏi train_loss chậm hơn (khoảng cách train-val nhỏ hơn = ít overfit hơn).
- **Lưu ý:** Dropout hoạt động khác nhau ở train và eval; nhớ net.train() trước khi train và net.eval() trước khi đánh giá (khoá học đánh giá val trong torch.no_grad()). Nếu quên eval(), dropout vẫn bật khi đo val -> số liệu nhiễu. Thử từng loại regularisation RIÊNG, đừng trộn.
- *Nguồn: Week3Exercise2_Percolation_CNN / CourseWorkProject*

#### Quá khớp (Overfitting)

- **Là gì:** Mô hình khớp rất tốt tập train (loss/error thấp) nhưng lại tệ hơn hẳn trên tập validation — nó đã 'học thuộc' cả nhiễu thay vì quy luật chung.
- **Ý nghĩa & mục đích:** Là vấn đề cốt lõi mà regularisation và thêm dữ liệu dùng để giải quyết. Trong đồ án, bước 1 CỐ TÌNH tạo mô hình đủ mạnh để overfit (train error dưới mục tiêu, val error cao hơn) làm điểm xuất phát cho các thí nghiệm.
- **Code:**

```python
# Ensure your model is complex enough to overfit the training data:
# training loss/error below target, validation loss/error higher.
# Symptom seen with a too-large Dense layer on percolation:
# fits training data but fails almost completely on validation data.
```
- **Visual:** Vẽ train_loss và val_loss theo epoch trên cùng đồ thị: dấu hiệu overfit là train_loss tiếp tục giảm còn val_loss chững lại rồi TĂNG, khoảng cách hai đường doãng ra. plt.plot(history['train_loss']); plt.plot(history['val_loss']); plt.legend().
- **Lưu ý:** Đừng nhìn mỗi train accuracy mà tưởng mô hình tốt — luôn so với validation. Model quá nhiều neuron dễ overfit; quá ít lại không khớp nổi train (underfit).
- *Nguồn: Week3Exercise2_Percolation_CNN / CourseWorkProject*

#### Thiếu khớp & Năng lực mô hình (Underfitting / capacity)

- **Là gì:** Underfitting: mô hình không khớp nổi ngay cả tập train vì thiếu 'năng lực' (capacity = độ linh hoạt: số neuron, số lớp) để biểu diễn hàm cần học.
- **Ý nghĩa & mục đích:** Dùng để chẩn đoán khi cả train lẫn val đều tệ: cần TĂNG năng lực (thêm neuron/lớp), không phải thêm regularisation. Ví dụ logistic regression (NN1) underfit hoàn toàn bài percolation vì hàm cần học phi tuyến mạnh.
- **Code:**

```python
# too few neurons -> cannot even fit the training data (underfit)
# fix: add more complexity - more neurons, more layers
class NN_one_hidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 100), nn.ReLU(),
            nn.Linear(100, 1), nn.Sigmoid())
    def forward(self, x):
        return self.layers(x)
```
- **Visual:** train_loss và val_loss đều cao và phẳng (không giảm xuống dưới mục tiêu) -> underfit. So với overfit (train thấp, val cao).
- **Lưu ý:** Regularise thêm khi đang underfit sẽ làm tệ hơn. Với percolation: L1/L2/dropout chỉ có ích SAU khi mô hình đã đủ mạnh để overfit. Chú ý giới hạn CPU: <100 neuron/lớp nối tiếp nhau (1000x1000 = 1 triệu weight là quá nhiều cho CPU).
- *Nguồn: Week3Exercise2_Percolation_CNN*

#### Khái quát hoá (Generalisation)

- **Là gì:** Khả năng mô hình dự đoán đúng trên dữ liệu MỚI chưa từng thấy, không chỉ trên dữ liệu đã học.
- **Ý nghĩa & mục đích:** Là mục tiêu thật sự của học máy. Vì không gian đầu vào quá lớn (8x8 nhị phân có 2^64 > 18 tỉ mẫu) nên không thể học thuộc — buộc mạng phải rút ra quy luật chung. Regularisation và tăng dữ liệu là hai cách chính để cải thiện generalisation.
- **Code:**

```python
# 2.0 ** 64  -> more than 18 billion 8x8 binary patterns
# training sets are far fewer, so the NN must generalise
```
- **Visual:** Learning curve: vẽ val error theo kích thước tập train (log-log) — đường đi xuống cho thấy generalisation cải thiện khi có thêm dữ liệu.
- **Lưu ý:** Train accuracy cao KHÔNG đảm bảo generalisation; luôn đo trên test set đủ lớn (>=10,000 mẫu để ước lượng error-rate đáng tin).
- *Nguồn: Week3Exercise2_Percolation_CNN / CourseWorkProject*

#### Đường mất mát train vs validation (Training vs Validation loss curves)

- **Là gì:** Đồ thị loss (và accuracy) theo từng epoch, vẽ riêng cho tập train và tập validation, lấy từ 'history' thu được khi train.
- **Ý nghĩa & mục đích:** Công cụ chẩn đoán trực quan số 1: nhìn khoảng cách và xu hướng hai đường để biết mô hình đang underfit, khớp tốt, hay overfit — từ đó quyết định thêm năng lực hay thêm regularisation.
- **Code:**

```python
history = {'train_loss': train_loss, 'train_acc': train_acc,
           'val_loss': val_loss, 'val_acc': val_acc}

plt.plot(history1['train_loss'], label='train_loss')
plt.plot(history1['val_loss'],   label='val_loss')
plt.legend()
```
- **Visual:** Hai đường trên cùng trục epoch. Khớp tốt: hai đường sát nhau và cùng giảm. Overfit: val chững/tăng trong khi train giảm. Underfit: cả hai cao. Vẽ tương tự cho accuracy.
- **Lưu ý:** Tính val bằng torch.no_grad() để khỏi tốn gradient và khỏi vô tình cập nhật trọng số. Nhớ lấy .item() cho tensor 0 chiều khi cộng dồn loss (total_loss += loss.item()).
- *Nguồn: Week3Exercise2_Percolation_CNN*

#### Quy tắc vàng khi tìm mô hình (Golden Rules of Model Search)

- **Là gì:** Quy trình lặp có kỷ luật để cải thiện mô hình: xác định error mục tiêu, khớp mô hình, rồi dựa vào train/val error để quyết định thêm năng lực hay thêm regularisation/dữ liệu.
- **Ý nghĩa & mục đích:** Dùng để tránh 'chỉnh mò' — thay đổi ngẫu nhiên rồi mong nó tốt lên. Đây là kim chỉ nam cho toàn bộ đồ án và mọi lần thiết kế kiến trúc mới.
- **Code:**

```python
# Step 1: decide the ideal error rate you expect a good model to reach.
# Step 2: fit a model; look at TRAIN and VALIDATION errors.
# Step 3a: doesn't fit train well -> too little capacity:
#          add neurons / layers, go to Step 2.
# Step 3b: fits train but higher val error -> overfitting:
#          use more training data AND/OR add regularisation, go to Step 2.
```
- **Visual:** Vẽ sơ đồ cây quyết định: 'Khớp train tốt?' — Không -> tăng năng lực; Có -> 'Val cao hơn?' — Có -> thêm dữ liệu/regularise; Không -> xong.
- **Lưu ý:** Đừng nhảy sang regularise khi mô hình còn chưa khớp nổi train (đó là thiếu năng lực, không phải overfit). Làm 'insightful', đừng random.
- *Nguồn: Week3Exercise2_Percolation_CNN*

#### Error mục tiêu / lý tưởng (Ideal error rate)

- **Là gì:** Mức lỗi mà bạn kỳ vọng một mô hình TỐT đạt được cho bài toán — có thể là 0, hoặc một mức dư không tránh khỏi (ngay cả con người cũng sai).
- **Ý nghĩa & mục đích:** Đặt trước làm mốc so sánh: nhờ nó mới biết train error đã 'đủ thấp' chưa (để phân biệt underfit vs overfit) và val error có chấp nhận được không. Là Bước 1 của Quy tắc vàng.
- **Code:**

```python
# Step 1 of model search:
# what error rate do you expect a good model to have?
# Call this the *ideal error rate*.
# (Percolation: nobody has reached 100% accuracy even after 30,000 examples.)
```
- **Visual:** Vẽ một đường ngang 'ideal error rate' trên đồ thị error theo epoch; xem train/val error nằm trên hay dưới đường đó để chẩn đoán.
- **Lưu ý:** Đặt mục tiêu phi thực tế (vd 0% cho bài rất khó như percolation) sẽ khiến bạn train vô ích. Một số bài có error dư không thể loại bỏ.
- *Nguồn: Week3Exercise2_Percolation_CNN*

#### Đường cong học & luật luỹ thừa (Learning curve / power law)

- **Là gì:** Đồ thị val error theo kích thước tập train, thường vẽ log-log; nếu error giảm theo một luỹ thừa của n thì trên trục log-log nó là một đường gần thẳng.
- **Ý nghĩa & mục đích:** Trả lời câu hỏi thực tế 'thêm dữ liệu có đáng không, và cải thiện nhanh cỡ nào?'. Đo độ dốc để ước lượng luật luỹ thừa (vd error ∝ 1/√n là rất tốt). Là thí nghiệm bắt buộc trong đồ án và là cách so sánh hai kiến trúc.
- **Code:**

```python
# error-rate vs training-set size, on a log-log plot:
plt.loglog([1000, 2000, 4000, 8000, 16000],
           [0.0636, 0.0330, 0.0130, 0.006, 0.0046])
# sizes at least 10x apart, e.g. 500, 1000, 2000, ..., 32000
```
- **Visual:** plt.loglog(train_sizes, val_errors): tìm đoạn thẳng -> độ dốc là số mũ của luật luỹ thừa. Vẽ hai đường (model 1 và model 2) cùng đồ thị để so sánh tốc độ cải thiện.
- **Lưu ý:** Cần GIỮ NGUYÊN mức regularisation đã chọn khi quét kích thước train. Test set phải đủ lớn (>=10,000) để error đo được không bị nhiễu. Không cần chạy quá lâu — không được điểm thêm vì thí nghiệm dài.
- *Nguồn: Week3Exercise2_Percolation_CNN / CourseWorkProject*

#### Bất ổn do trọng số lớn (Instability from large weights)

- **Là gì:** Trong mô hình 'sâu' đơn giản nhất ŷ=w₁w₂x, gradient theo w₁ tỉ lệ với w₂ (và ngược lại); nếu một trọng số phình to thì gradient của trọng số kia bùng nổ, khiến học phân kỳ dù mô hình toán học vẫn tương đương ŷ=wx.
- **Ý nghĩa & mục đích:** Đây là ví dụ minh hoạ TẠI SAO cần regularisation: có cả một họ nghiệm 'tốt như nhau' (mọi w₁,w₂ với w₁w₂=1) nhưng nhiều nghiệm có trọng số cực đoan gây bất ổn số học. Regularisation (L2) loại bỏ các nghiệm cực đoan đó.
- **Code:**

```python
w1 = 0.7; w2 = 1.0/0.7   # w1*w2 == 1
learning_rate = 0.01
for x, y in zip(xx, yy):
    yhat = w1 * w2 * x
    grad = (yhat - y) * x
    w1 = w1 - learning_rate * grad * w2
    w2 = w2 - learning_rate * grad * w1
# without regularisation, w heads off to infinity (goes 'wild')
```
- **Visual:** plt.plot(np.array(ww1[:limit]) * np.array(ww2[:limit])) — tích w1*w2 lẽ ra luôn =1 nhưng bạn thấy nó phình dần rồi bùng nổ. Hoặc plt.plot(ww1[:limit], ww2[:limit]) để thấy quỹ đạo bắn ra vô cực.
- **Lưu ý:** Learning rate quá cao -> phân kỳ nhanh hơn; hãy dò 'limit' bằng thử-sai để thấy đúng lúc trước khi w bay đi vô cực. Mô hình 'sâu' hơn (nhân nhiều weight) làm bất ổn này tệ hơn.
- *Nguồn: Week1Exercise2_SimpleDNN*

#### Đánh đổi khi regularise (Regularisation trade-off)

- **Là gì:** Regularisation càng mạnh thì trọng số càng bị ép nhỏ; đến mức nào đó nó ép cả những trọng số CẦN THIẾT về 0, làm dự đoán lệch đi — nên luôn có một mức 'tối ưu' ở giữa.
- **Ý nghĩa & mục đích:** Giúp hiểu vì sao đồ thị 'val performance theo mức regularisation' có hình chữ U và vì sao phải quét nhiều mức để tìm điểm tối ưu, thay vì cứ tăng regularisation vô tội vạ.
- **Code:**

```python
# also plot yy versus yhats: what happens when the regulariser is too big?
# What is the compromise involved in introducing regularisation?
# A small amount may improve validation; too much makes it worse.
```
- **Visual:** Vẽ val_loss (hoặc error) theo mức regularisation trên trục log x -> đường chữ U: đáy = mức tối ưu. Kèm scatter yy vs yhats khi regulariser quá lớn để thấy dự đoán bị 'kéo dẹt'.
- **Lưu ý:** Mức tối ưu là điểm cân bằng, không phải 'càng nhiều càng tốt'. Với L1/L2 vùng ~0.00001–0.001; vượt ngưỡng thì mọi trọng số bị đẩy về 0 và mô hình vô dụng.
- *Nguồn: Week1Exercise2_SimpleDNN / CourseWorkProject*

#### Tập kiểm định để đánh giá trung thực (Validation / test set)

- **Là gì:** Một tập dữ liệu tách riêng, KHÔNG dùng để cập nhật trọng số, chỉ dùng để đo loss/accuracy nhằm ước lượng khả năng generalisation thật của mô hình.
- **Ý nghĩa & mục đích:** Là 'sự thật' để phát hiện overfit và để chọn mức regularisation / kích thước mô hình. Mọi quyết định (thêm neuron, thêm regularisation) đều dựa trên val chứ không dựa trên train.
- **Code:**

```python
def accuracy_and_loss(net, loss_function, dataloader):
    total_correct = 0; total_loss = 0.0; total_examples = 0; n_batches = 0
    with torch.no_grad():           # no gradients on the validation set
        for images, labels in dataloader:
            outputs = net(images)
            total_loss += loss_function(outputs, labels).item()
            total_correct += sum((outputs > 0.5) == (labels > 0.5)).item()
            total_examples += labels.size(0); n_batches += 1
    return (total_correct / total_examples, total_loss / n_batches)
```
- **Visual:** Luôn vẽ đường val song song với train trên mọi đồ thị loss/accuracy; khoảng cách giữa chúng chính là mức overfit.
- **Lưu ý:** Bọc bằng torch.no_grad() để không tính/tích luỹ gradient trên val. Test set phải đủ lớn (>=10,000) để con số đáng tin. Đừng bao giờ dùng val để cập nhật trọng số. Deprecation/best-practice: dùng nn.BCELoss() sau sigmoid vẫn chạy, nhưng nn.BCEWithLogitsLoss() (bỏ lớp Sigmoid cuối, đưa logit vào) ổn định số hơn.
- *Nguồn: Week3Exercise2_Percolation_CNN / CourseWorkProject*

#### Soi lỗi để hiểu mô hình (Inspection of errors: false positives/negatives)

- **Là gì:** Xem lại cụ thể các mẫu mà mô hình dự đoán sai trên tập train và val, tách thành false-negative và false-positive, để tìm quy luật mô hình đang làm sai.
- **Ý nghĩa & mục đích:** Bổ trợ cho các con số tổng quát (accuracy/loss): giúp hiểu ĐỊNH TÍNH điểm yếu của mô hình và gợi ý cách cải thiện kiến trúc — bước 'devise a better model' trong đồ án.
- **Code:**

```python
# find the errors on the training and validation sets, and inspect them.
# false-negatives (percolating but predicted not) tended to have long paths
# false-positives showed no obvious regularity
```
- **Visual:** Hiển thị các ảnh sai bằng plt.imshow(x.numpy().squeeze()); nhóm theo false-neg / false-pos rồi tìm đặc điểm chung (vd đường percolate dài). Có thể dùng bộ 'corner cases' (test_long_path, test_row_with_gap, ...) để dò lỗi hệ thống.
- **Lưu ý:** Đừng chỉ nhìn accuracy tổng: hai mô hình cùng accuracy có thể sai ở kiểu mẫu rất khác nhau. Nhớ squeeze() để bỏ chiều kênh khi vẽ ảnh.
- *Nguồn: Week3Exercise2_Percolation_CNN*

#### Lấy minibatch từ DataLoader (DataLoader / minibatch iteration)

- **Là gì:** DataLoader là tiện ích PyTorch bọc quanh dataset để lặp qua từng minibatch (đã shuffle, đã gộp thành tensor theo batch), phục vụ cả huấn luyện lẫn đánh giá val.
- **Ý nghĩa & mục đích:** Là 'ống dẫn dữ liệu' chuẩn cho vòng lặp train/val. Đổi batch_size chỉ cần tạo DataLoader mới; nó cũng là chỗ để soi thử một batch (kiểm tra shape, nhãn) trước khi train — thói quen 'always check the shapes'.
- **Code:**

```python
trainloader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_set,     batch_size=32, shuffle=True)

# peek at one minibatch:
tmpiter = iter(trainloader)
images, labels = next(tmpiter)   # modern API
images.shape                     # always check tensor shapes
```
- **Visual:** In images.shape ra để xác nhận dạng [batch, channel, H, W]; vẽ một phần tử bằng plt.imshow(images[0].numpy().squeeze()) kèm labels[0] để chắc dữ liệu và nhãn khớp nhau.
- **Lưu ý:** DEPRECATION: cú pháp cũ tmpiter.next() (như trong notebook) đã bị bỏ — dùng next(tmpiter). Đổi batch_size phải tạo DataLoader mới, không sửa được tại chỗ. shuffle=True cho train; với val thường để False cũng được vì không cập nhật trọng số.
- *Nguồn: Week3Exercise2_Percolation_CNN*


---

### 8.11 Kỹ thuật Visualize

#### Bật vẽ hình trong notebook (%matplotlib inline)

- **Là gì:** Lệnh magic của Jupyter để hình vẽ matplotlib hiện ngay trong notebook, kèm import chuẩn plt.
- **Ý nghĩa & mục đích:** Phải chạy một lần đầu notebook, nếu không hình sẽ không hiện (hoặc bật ra cửa sổ riêng). Dùng bí danh plt cho gọn thay vì gõ matplotlib.pyplot mỗi lần.
- **Code:**

```python
import matplotlib.pyplot as plt
%matplotlib inline
```
- **Visual:** Không có hình riêng; đây là bước chuẩn bị để mọi plt.plot / plt.imshow sau đó render inline.
- **Lưu ý:** Là magic của Jupyter, KHÔNG chạy được trong file .py thường. Có notebook dùng %pylab inline (kiểu cũ, đổ cả numpy+pyplot vào namespace) — nay khuyến nghị dùng %matplotlib inline + import tường minh.
- *Nguồn: Week1Exercise1_GradientDescent, Week1Exercise3_NN_Backpropagation*

#### Vẽ đường / rải điểm (plt.plot với format string)

- **Là gì:** plt.plot vẽ đường nối các điểm; thêm chuỗi định dạng như 'r.' để đổi thành rải điểm màu.
- **Ý nghĩa & mục đích:** Xem nhanh một vector số (đường cong), hoặc rải cặp (x,y) để nhìn phân bố dữ liệu. Cách nhanh nhất để 'nhìn' xem chuyện gì đang xảy ra với mảng số.
- **Code:**

```python
plt.plot(v)              # đường cong từ 1 vector
plt.plot(x, y, 'r.')     # rải điểm màu đỏ
plt.plot(x, y, '.')      # rải điểm mặc định
```
- **Visual:** Trục hoành = chỉ số (hoặc x), trục tung = giá trị. 'r.'=chấm đỏ, 'b.'=chấm xanh, 'bo'=chấm tròn xanh, 'b'=đường liền.
- **Lưu ý:** Format string ngắn dễ nhầm: 'r' là đường đỏ liền, 'r.' mới là chấm đỏ. Với tensor phải .detach().cpu().numpy() trước khi vẽ.
- *Nguồn: Week1Exercise1_GradientDescent*

#### Vẽ chồng nhiều lớp trên cùng một trục (overlay plots)

- **Là gì:** Gọi nhiều plt.plot liên tiếp thì chúng vẽ đè lên cùng một hình cho đến khi sang cell/figure mới.
- **Ý nghĩa & mục đích:** Dùng để so sánh trực tiếp: vẽ hai nhóm dữ liệu khác màu, hoặc phủ đường/điểm lên nền colormap. Rất hay để tách 2 lớp theo nhãn.
- **Code:**

```python
plt.plot(X[Y==0.0,0], X[Y==0.0,1], 'b.')   # lớp 0 xanh
plt.plot(X[Y==1.0,0], X[Y==1.0,1], 'r.')   # lớp 1 đỏ
```
- **Visual:** Hai đám điểm hai màu trên cùng khung; boolean mask Y==0.0 chọn các điểm của một lớp.
- **Lưu ý:** Muốn tách thành hình riêng phải gọi plt.figure() ở giữa, nếu không mọi thứ dồn vào một hình.
- *Nguồn: Week1Exercise3_NN_Backpropagation, Week2Exercise2_Backpropagation_with_pytorch*

#### Hiện ma trận thành bản đồ màu (plt.imshow + colorbar)

- **Là gì:** imshow vẽ một ma trận 2D thành lưới ô màu; colorbar thêm thang màu bên cạnh để đọc giá trị.
- **Ý nghĩa & mục đích:** Cách chính để 'nhìn' nội dung một ma trận lớn mà mắt không đọc số được — thấy vùng nào cao/thấp, cấu trúc, nhiễu. Nền tảng cho mọi heatmap về sau.
- **Code:**

```python
plt.imshow(a)
plt.colorbar()
```
- **Visual:** Lưới ô, màu = giá trị ô; colorbar cho biết màu nào ứng giá trị nào.
- **Lưu ý:** Không có colorbar thì không biết thang giá trị. Với ma trận rất lệch (một ô cực lớn) màu bị 'nuốt' — cân nhắc scale lại (xem thẻ 1/J).
- *Nguồn: Week1Exercise1_GradientDescent*

#### Hiển thị ảnh RGB (imshow + permute kênh màu)

- **Là gì:** Dùng imshow để hiện ảnh màu, nhưng phải đưa tensor về dạng (H, W, C) mà matplotlib mong đợi.
- **Ý nghĩa & mục đích:** PyTorch lưu ảnh dạng (C, H, W) còn matplotlib cần (H, W, C), nên phải hoán vị trục trước khi hiện, nếu không ảnh sẽ méo/sai màu.
- **Code:**

```python
plt.imshow(input_img.permute(1,2,0).cpu().numpy())
```
- **Visual:** Ảnh màu 32x32 (CIFAR-10) hiện đúng; permute(1,2,0) chuyển (C,H,W)→(H,W,C).
- **Lưu ý:** Quên permute → lỗi shape hoặc ảnh loạn. Tensor trên GPU phải .cpu() trước .numpy(). Giá trị pixel nên nằm trong [0,1] (float) hoặc [0,255] (uint8).
- *Nguồn: visualize_activation (visualize_filters_for_class)*

#### Hiện ảnh xám / ma trận 1 kênh bằng squeeze (imshow + .squeeze())

- **Là gì:** Bỏ chiều kênh thừa (size 1) của tensor/mảng bằng .squeeze() để imshow hiện được ảnh xám 2D.
- **Ý nghĩa & mục đích:** Ảnh đen trắng thường lưu dạng (1,H,W) hoặc (H,W,1); nhưng imshow ảnh xám cần mảng 2D (H,W). squeeze() bỏ chiều size-1 để hiện đúng, dùng để soi nhanh một mẫu dữ liệu (vd ảnh percolation 8x8 nhị phân) xem nhãn có khớp không.
- **Code:**

```python
x, y = training_set[376]
plt.imshow(x.numpy().squeeze())   # (1,8,8) -> (8,8)
plt.colorbar()
```
- **Visual:** Ảnh 2D (vd lưới 8x8 nhị phân), mỗi ô = 0/1; squeeze bỏ chiều kênh để khỏi lỗi shape khi imshow.
- **Lưu ý:** Chỉ squeeze được chiều bằng 1. Ảnh nhiều kênh (RGB) KHÔNG squeeze về 2D được — phải permute (xem thẻ RGB). imshow ảnh 2D mặc định dùng colormap giả màu (viridis), muốn xám thật thêm cmap='gray'.
- *Nguồn: Week3Exercise2_Percolation_CNN*

#### Điều chỉnh trục cho imshow (origin='lower', extent)

- **Là gì:** origin='lower' đặt gốc ở góc dưới-trái (kiểu toán học); extent=[xmin,xmax,ymin,ymax] gán nhãn giá trị thật cho trục thay vì chỉ số ô.
- **Ý nghĩa & mục đích:** Khi ma trận biểu diễn một hàm trên miền tham số (ví dụ J theo m và c), cần trục hiển thị đúng giá trị tham số và gốc đúng chiều, không phải chỉ số pixel.
- **Code:**

```python
plt.imshow(J_grid.transpose(), origin='lower', extent=[0,1,0,2])
plt.colorbar()
```
- **Visual:** Bản đồ màu của J với trục x=m∈[0,1], y=c∈[0,2]; transpose để trục đúng chiều.
- **Lưu ý:** Mặc định imshow đặt gốc ở góc trên-trái và đảo trục y → dễ đọc ngược. Thường phải .transpose() ma trận cho khớp ý nghĩa trục.
- *Nguồn: Week1Exercise1_GradientDescent*

#### Vẽ mặt lỗi bằng quét lưới tham số (error surface / J_grid)

- **Là gì:** Tính giá trị hàm mất mát J tại từng ô của lưới 2 tham số rồi imshow — cho ra 'bản đồ' mặt lỗi.
- **Ý nghĩa & mục đích:** Với bài 2 tham số, đây là cách nhìn thấy toàn cảnh mặt lỗi: đâu là đáy (nghiệm tốt), lòng chảo hẹp hay rộng, để hiểu vì sao gradient descent đi kiểu này.
- **Code:**

```python
J_grid = np.zeros([m_values.size, c_values.size])
for i in range(m_values.size):
    for j in range(c_values.size):
        J_grid[i,j] = calculate_J(x_data, y_data, m_values[i], c_values[j])
plt.imshow(J_grid.transpose(), origin='lower', extent=[0,1,0,2])
plt.colorbar()
```
- **Visual:** Colormap dạng lòng chảo; vùng tối = J nhỏ (đáy). Chỉ khả thi khi có đúng 2 tham số để quét.
- **Lưu ý:** Chỉ dùng được cho 2 chiều (mạng thật quá nhiều chiều nên không quét lưới được). Vòng lặp đôi chậm nếu lưới lớn.
- *Nguồn: Week1Exercise1_GradientDescent*

#### Mẹo nghịch đảo để dễ đọc colormap (1/J)

- **Là gì:** Khi mặt lỗi có vùng giá trị rất lớn nuốt hết màu, vẽ 1/J thay vì J để làm nổi vùng đáy.
- **Ý nghĩa & mục đích:** Đáy lòng chảo (J nhỏ) là chỗ ta quan tâm nhưng lại bị màu nền lấn át; lấy nghịch đảo biến đáy thành đỉnh sáng, dễ nhìn vị trí nghiệm.
- **Code:**

```python
plt.imshow(1. / J_grid.transpose(), origin='lower', extent=[0,1,0,2])
plt.colorbar()
```
- **Visual:** Cùng mặt lỗi nhưng vùng nghiệm tối ưu giờ sáng rực (giá trị 1/J lớn).
- **Lưu ý:** Coi chừng chia cho 0 nếu J có ô bằng 0. Đây là mẹo trực quan, không đổi bản chất bài toán.
- *Nguồn: Week1Exercise1_GradientDescent*

#### Vẽ đường đi của gradient descent trên mặt lỗi

- **Là gì:** Ghi lại (m,c) qua từng bước cập nhật rồi phủ đường + điểm lên bản đồ mặt lỗi.
- **Ý nghĩa & mục đích:** Nhìn tận mắt thuật toán 'lăn' xuống đáy thế nào: có zigzag không, hội tụ hay phân kỳ, nhạy với learning rate ra sao. Rất trực quan cho việc dạy tối ưu.
- **Code:**

```python
m_path=[m]; c_path=[c]
for n in range(n_iterations):
    m_grad, c_grad = J_gradient(x_data, y_data, m, c)
    m -= learning_rate*m_grad; c -= learning_rate*c_grad
    m_path.append(m); c_path.append(c)
plt.imshow(1./J_grid.transpose(), origin='lower', extent=[0,1,0,2]); plt.colorbar()
plt.plot(m_path, c_path, 'r'); plt.plot(m_path, c_path, 'r.')
```
- **Visual:** Đường đỏ zigzag men theo lòng chảo tới đáy; mỗi chấm là một bước.
- **Lưu ý:** Learning rate quá cao → đường bật ra ngoài, phân kỳ (giá trị path phình to). Nếu lòng chảo hẹp một chiều, đường đi dao động mạnh theo chiều dốc.
- *Nguồn: Week1Exercise1_GradientDescent*

#### Vẽ mặt dự đoán / ranh giới quyết định của mạng (plot_nn_predictions)

- **Là gì:** Quét lưới điểm 2D, cho mạng dự đoán từng điểm, imshow kết quả rồi phủ dữ liệu huấn luyện lên.
- **Ý nghĩa & mục đích:** Với input 2 chiều, đây là cách thấy mạng đã học ranh giới phân lớp thế nào và có khớp dữ liệu không. Lý do dùng bài toán 2D là để trực quan hoá được (mạng thật nhiều chiều thì chịu).
- **Code:**

```python
def plot_nn_predictions(net1):
    xs = np.linspace(-1,1,200); ys = np.linspace(-1,1,200)
    J_grid = np.zeros([xs.size, ys.size])
    with torch.no_grad():
        for i in range(xs.size):
            for j in range(ys.size):
                J_grid[i,j] = net1(torch.FloatTensor([xs[i], ys[j]])).item()
    plt.imshow(J_grid.transpose(), origin='lower', extent=[-1,1,-1,1]); plt.colorbar()
plot_nn_predictions(net1)
plt.plot(X[Y==0.0,0], X[Y==0.0,1], 'b.')
plt.plot(X[Y==1.0,0], X[Y==1.0,1], 'r.')
```
- **Visual:** Nền màu = xác suất/đầu ra mạng theo vùng; chấm xanh/đỏ = dữ liệu 2 lớp phủ lên để kiểm tra ranh giới.
- **Lưu ý:** Dự đoán từng điểm một rất chậm (200x200 lần forward) — có thể gộp batch. Nhớ bọc torch.no_grad() để khỏi tốn bộ nhớ tính gradient.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch, Week1Exercise3_NN_Backpropagation*

#### Đường cong loss/accuracy theo epoch (train vs val từ history)

- **Là gì:** Lưu loss/accuracy mỗi epoch vào dict history rồi vẽ hai đường train và validation trên cùng hình.
- **Ý nghĩa & mục đích:** Công cụ chẩn đoán cốt lõi: nhìn khoảng cách train vs val để biết mạng đang underfit (cả hai kém) hay overfit (train tốt, val kém), và khi nào nên dừng.
- **Code:**

```python
plt.plot(history1['train_loss'], label='train_loss')
plt.plot(history1['val_loss'], label='val_loss')
plt.legend()
```
- **Visual:** Hai đường theo epoch: train_loss giảm dần; nếu val_loss quay đầu đi lên trong khi train vẫn giảm → overfit.
- **Lưu ý:** Không gọi plt.legend() thì không biết đường nào là gì. Vẽ chung loss và acc dễ rối vì thang khác nhau — tách hình bằng plt.figure().
- *Nguồn: Week3Exercise2_Percolation_CNN, 1 - What_can_recurrent_nets_learn*

#### Tách hình và chú thích (plt.figure, label, legend, title)

- **Là gì:** plt.figure() mở một khung vẽ mới; label trong plot + plt.legend() tạo chú thích; plt.title đặt tiêu đề.
- **Ý nghĩa & mục đích:** Khi cần nhiều biểu đồ riêng (ví dụ accuracy và loss tách nhau) và cần ghi rõ đường nào là train/val để hình tự giải thích được.
- **Code:**

```python
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy'); plt.legend()
plt.figure()   # sang hình mới cho loss
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss'); plt.legend()
plt.show()
```
- **Visual:** Hai hình riêng: một cho accuracy, một cho loss; 'bo'=chấm xanh (train), 'b'=đường xanh (val).
- **Lưu ý:** Quên plt.figure() → loss vẽ đè lên hình accuracy. plt.show() cần khi không ở chế độ inline tự hiện.
- *Nguồn: 1 - What_can_recurrent_nets_learn*

#### Đồ thị log-log để tìm luật lũy thừa (plt.loglog)

- **Là gì:** plt.loglog vẽ cả hai trục theo thang log; luật lũy thừa (power law) hiện thành đường gần thẳng.
- **Ý nghĩa & mục đích:** Dùng để khảo sát quan hệ như 'sai số validation giảm thế nào theo kích thước tập huấn luyện': nếu là luật lũy thừa thì trên log-log nó thành đường thẳng, đọc được số mũ.
- **Code:**

```python
plt.loglog([1000,2000,4000,8000,16000],
           [0.0636,0.0330,0.0130,0.006,0.0046])
# hoặc để xem tốc độ hội tụ của J:
plt.loglog(global_mean_J)
```
- **Visual:** Cả hai trục log; dữ liệu tuân theo luật lũy thừa nằm gần một đường thẳng, độ dốc = số mũ.
- **Lưu ý:** loglog không nhận giá trị ≤ 0 (log âm vô nghĩa) — ví dụ vẽ độ giảm dùng -np.diff(...) để lấy phần dương.
- *Nguồn: Week3Exercise2_Percolation_CNN, Week1Exercise3_NN_Backpropagation*

#### Lưới nhiều ảnh (plt.subplots + ax.flatten + axis off)

- **Là gì:** Tạo lưới nhiều ô con bằng plt.subplots, làm phẳng mảng trục, rồi imshow từng ảnh và tắt trục.
- **Ý nghĩa & mục đích:** Khi cần xem nhiều ảnh cùng lúc (mẫu mỗi lớp, các feature map, các bộ lọc) trong một khung gọn gàng, có tiêu đề, không có khung trục rối mắt.
- **Code:**

```python
fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(18,18))
ax = ax.flatten()
for i in range(n_images):
    ax[i].imshow(images_to_plot[i])
    ax[i].axis('off')
    if titles is not None and i < 10:
        ax[i].set_title(titles[i%10])
```
- **Visual:** Bảng ảnh nhiều hàng nhiều cột; mỗi ô một ảnh, tắt trục, có thể gắn tên lớp làm tiêu đề.
- **Lưu ý:** ax là mảng 2D khi có nhiều hàng/cột — phải .flatten() để index 1 chiều. Số ô trống dư (n_images < hàng*cột) sẽ hiện ô trắng. Lưu ý: cách lấy batch ảnh để vẽ bằng dataiter.next() trong notebook nay đã deprecated → dùng next(dataiter).
- *Nguồn: visualize_activation (plot_colour_images, visualize_filters_for_class)*

#### Histogram trọng số và gradient (plt.hist)

- **Là gì:** plt.hist vẽ phân bố tần suất giá trị; áp lên mảng trọng số hoặc gradient đã trải phẳng.
- **Ý nghĩa & mục đích:** Nhìn phân bố trọng số/gradient để chẩn đoán: gradient dồn quanh 0 (vanishing), trọng số phình quá to, hay đối xứng lành mạnh. Cách nhanh soi 'sức khoẻ' của mạng.
- **Code:**

```python
layer1_weights = net1.layer1.weight.detach().numpy()
layer1_gradients = net1.layer1.weight.grad.numpy()
n_bins = 7
plt.hist(layer1_weights.reshape(-1), n_bins)
plt.hist(layer1_gradients.reshape(-1), n_bins)
```
- **Visual:** Trục hoành = độ lớn giá trị, trục tung = số lượng trọng số/gradient rơi vào khoảng đó; hình chuông quanh 0 là bình thường.
- **Lưu ý:** Phải .reshape(-1) trải ma trận thành vector 1D. Trọng số cần .detach().numpy() (vì có gradient bám), .grad thì là tensor thuần nên .numpy() trực tiếp được. Nếu tensor trên GPU phải thêm .cpu() trước .numpy().
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch*

#### Heatmap trọng số & bias của một lớp

- **Là gì:** Xếp bias và ma trận trọng số của một lớp thành một ma trận rồi imshow để nhìn từng neuron.
- **Ý nghĩa & mục đích:** Thấy trực quan neuron nào có trọng số mạnh/yếu, có neuron nào 'chết' (toàn 0, không được dùng) không. Cách soi cấu trúc học được của lớp.
- **Code:**

```python
plt.imshow(np.vstack([nn.bias_1, nn.w_1]))
plt.colorbar()
```
- **Visual:** Bản đồ màu: mỗi cột ~ một neuron, mỗi hàng ~ một trọng số/bias; ô sáng/tối = giá trị lớn/nhỏ.
- **Lưu ý:** Muốn theo dõi gradient qua thời gian: copy ma trận gradient mỗi bước, reshape(1,-1), rồi np.vstack danh sách lại thành heatmap. Nhớ COPY vì tensor gradient bị ghi đè mỗi bước.
- *Nguồn: Week1Exercise3_NN_Backpropagation*

#### Heatmap đầu ra LSTM dọc theo chuỗi

- **Là gì:** Cho một chuỗi qua mô hình LSTM 'phiên bản trực quan' (trả toàn bộ output mỗi bước), rồi imshow ma trận output theo vị trí ký tự.
- **Ý nghĩa & mục đích:** Nhìn xem trạng thái ẩn của LSTM thay đổi thế nào khi đọc từng ký tự — neuron nào 'bật' khi gặp ký tự quan trọng, cách mạng nhớ thông tin qua chuỗi.
- **Code:**

```python
lstm_outputs, h = model_viz(viz_vector)
plt.imshow(lstm_outputs[0,:,:].cpu().detach().numpy().transpose())
plt.colorbar()
```
- **Visual:** Heatmap: trục x = vị trí ký tự trong chuỗi, trục y = các neuron ẩn; màu = mức kích hoạt. Đổi một ký tự rồi so heatmap để thấy neuron nào phản ứng.
- **Lưu ý:** Phải chép trọng số sang một model chỉ-trả-output-đầy-đủ (model_viz) để lấy output mọi bước thay vì một nhãn yes/no; chép bằng load_state_dict (đúng cách khuyến nghị, thay vì save/load cả model). Nhớ .cpu().detach().numpy() và .transpose() cho đúng chiều trục.
- *Nguồn: 1 - What_can_recurrent_nets_learn*

#### Bắt output tầng bằng forward hook (get_activation)

- **Là gì:** Đăng ký một hook vào tầng; mỗi lần forward qua tầng đó, hook lưu output vào một dict để dùng sau.
- **Ý nghĩa & mục đích:** Muốn trực quan hoá activation của tầng giữa mạng nhưng forward chỉ trả kết quả cuối. Hook cho phép 'móc' lấy output tầng bất kỳ mà không sửa hàm forward.
- **Code:**

```python
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = F.relu(output)
    return hook
model.conv4.register_forward_hook(get_activation('conv4'))
# sau forward: layer_output = activation['conv4']
```
- **Visual:** Không phải hình; đây là cơ chế lấy dữ liệu. Output lấy được thường đem imshow thành feature map hoặc dùng làm loss cho activation maximization.
- **Lưu ý:** Hook chỉ chạy khi có forward pass mới; activation là dict được ghi đè mỗi lần forward. Nhớ gỡ hook (handle.remove()) nếu không còn cần, tránh giữ tham chiếu/rò bộ nhớ.
- *Nguồn: visualize_activation*

#### Sinh ảnh tối đa hoá kích hoạt bộ lọc (activation maximization)

- **Là gì:** Bắt đầu từ ảnh nhiễu, dùng gradient ASCENT để chỉnh ảnh sao cho output của một bộ lọc/neuron lớn nhất — ra 'ảnh mà bộ lọc thích nhất'.
- **Ý nghĩa & mục đích:** Trả lời câu hỏi 'bộ lọc này học mẫu gì?': thay vì xem trọng số, ta dựng ảnh input làm nó kích hoạt mạnh nhất, thấy được template (đường, góc, mắt...) mà nó dò.
- **Code:**

```python
input_img = torch.tensor(np.random.random((1,3,32,32))*20+128.,
                         dtype=torch.float32, device=device, requires_grad=True)
for _ in range(150):
    model(input_img)
    layer_output = activation[layer_name]
    loss_value = torch.mean(layer_output[:, filter_index, :, :])
    grads = torch.autograd.grad(loss_value, input_img)[0]
    grads /= (torch.sqrt(torch.mean(torch.square(grads))) + 1e-8)  # chuẩn hoá gradient
    with torch.no_grad():
        input_img += grads * step
```
- **Visual:** Kết quả là ảnh loang màu/hoạ tiết thể hiện mẫu bộ lọc thích; tầng sâu (conv4) cho mẫu phức tạp hơn tầng nông.
- **Lưu ý:** Đây là gradient ASCENT (cộng gradient, không trừ). Phải chuẩn hoá gradient để bước đi ổn định. Một số bộ lọc kẹt ở 0 (loss≈0) nên bỏ qua. Bọc cập nhật trong torch.no_grad(). Lưu ý: notebook gốc dùng torch.autograd.Variable — API này đã DEPRECATED, nay tạo tensor trực tiếp với requires_grad=True (như code trên), không cần Variable.
- *Nguồn: visualize_activation*

#### Chuẩn hoá tensor về ảnh hợp lệ (deprocess_image)

- **Là gì:** Đưa tensor giá trị tuỳ ý về khoảng pixel [0,255] uint8: căn giữa 0, chuẩn hoá độ lệch, dời về [0,1], clip rồi nhân 255.
- **Ý nghĩa & mục đích:** Ảnh sinh ra từ gradient ascent có giá trị lung tung; muốn hiện được bằng imshow phải ép về dải ảnh chuẩn, nếu không sẽ tràn màu hoặc toàn đen/trắng.
- **Code:**

```python
def deprocess_image(x):
    x -= torch.mean(x); x /= (torch.std(x)+1e-5); x *= 0.1
    x += 0.5
    x = np.clip(x.detach().cpu().numpy(), 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
```
- **Visual:** Biến tensor 'thô' thành ảnh RGB nhìn được; +1e-5 tránh chia cho 0 khi std=0.
- **Lưu ý:** Bước clip là bắt buộc, nếu không giá trị âm/tràn sẽ hiện sai. Thứ tự trục kênh phải khớp với cách imshow đọc (H,W,C).
- *Nguồn: visualize_activation*

#### Ghép nhiều bộ lọc thành một ảnh lưới và lưu file (draw_filters)

- **Là gì:** Xếp các ảnh bộ lọc đã sinh vào một lưới nxn có lề, chọn top theo loss cao, rồi lưu ra PNG bằng mpimg.imsave.
- **Ý nghĩa & mục đích:** Sau khi sinh ảnh cho nhiều bộ lọc, cần gom lại thành một hình tổng để so sánh và lưu ra đĩa; chọn các bộ lọc kích hoạt mạnh nhất (loss cao) vì thường 'đẹp' và dễ đọc mẫu hơn.
- **Code:**

```python
filters.sort(key=lambda x: x[1], reverse=True)   # loss cao lên trước
filters = filters[:n*n]
stitched = np.zeros((3, width, height), dtype='uint8')
# ... đặt từng img vào ô lưới với MARGIN ...
stitched = np.transpose(stitched, (1,2,0))
mpimg.imsave('{}_{}_{}x{}.png'.format(model_name, layer_name, n, n), stitched)
```
- **Visual:** Một ảnh lớn dạng lưới các ô mẫu bộ lọc, cách nhau bởi lề đen 5px; đọc từ file lại bằng mpimg.imread rồi plt.imshow.
- **Lưu ý:** stitched dựng theo trục (C,W,H) nên phải np.transpose về (H,W,C) trước khi lưu. Sắp xếp theo loss là quy ước 'bộ lọc kích hoạt mạnh = trực quan hơn', không phải chân lý.
- *Nguồn: visualize_activation*

#### Xem top feature map kích hoạt mạnh nhất cho một ảnh (visualize_filters_for_class)

- **Là gì:** Cho một ảnh thuộc lớp chọn qua mạng, lấy output các tầng conv qua hook, xếp hạng feature map theo activation trung bình, rồi imshow top 10 mỗi tầng.
- **Ý nghĩa & mục đích:** Thấy 'mỗi tầng nhìn ảnh này ra sao': tầng nông giữ nét gần ảnh gốc, tầng sâu thành các mẫu trừu tượng. Cách hiểu CNN xử lý một ảnh cụ thể qua từng tầng.
- **Code:**

```python
for layer in ['conv1','conv2','conv3','conv4']:
    output_images = np.squeeze(activation[layer], axis=0)
    filter_images = [(output_images[k,:,:], torch.mean(output_images[k,:,:]))
                     for k in range(output_images.shape[0])]
    filter_images.sort(key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(1, 10, figsize=(18,18)); ax = ax.flatten()
    for i, img in enumerate(filter_images[:10]):
        ax[i].imshow(img[0].detach().cpu().numpy()); ax[i].axis('off')
```
- **Visual:** Mỗi tầng một hàng 10 feature map kích hoạt mạnh nhất; càng sâu càng trừu tượng, độ phân giải giảm dần.
- **Lưu ý:** Phải chạy forward ảnh trước để hook điền activation. np.squeeze(axis=0) bỏ chiều batch=1 → còn (C,H,W), nên duyệt bộ lọc theo shape[0] (số kênh). Ảnh CIFAR-10 độ phân giải thấp nên mẫu không đẹp bằng mạng ImageNet.
- *Nguồn: visualize_activation*

#### Đưa tensor về dạng vẽ được (detach / cpu / numpy)

- **Là gì:** Chuỗi .detach().cpu().numpy() gỡ tensor khỏi đồ thị gradient, chuyển từ GPU về CPU, rồi thành mảng numpy để matplotlib vẽ.
- **Ý nghĩa & mục đích:** matplotlib chỉ hiểu numpy trên CPU; tensor có gradient hoặc nằm trên GPU sẽ báo lỗi khi plot. Đây là bước chuyển bắt buộc trước mọi imshow/plot với tensor PyTorch.
- **Code:**

```python
layer1_weights = net1.layer1.weight.detach().numpy()   # có gradient → cần detach
image = img.detach().cpu().numpy()                      # trên GPU → thêm .cpu()
```
- **Visual:** Không có hình; là bước chuyển dữ liệu. Nếu tensor là .grad thuần thì .numpy() trực tiếp, khỏi detach.
- **Lưu ý:** Quên detach với tensor requires_grad → RuntimeError. Quên .cpu() với tensor GPU → lỗi 'can't convert cuda tensor'. Thứ tự: detach trước, cpu sau, numpy cuối.
- *Nguồn: Week2Exercise2_Backpropagation_with_pytorch, visualize_activation*

#### Chỉnh kích thước hình (plt.rcParams figure.figsize)

- **Là gì:** Đặt kích thước mặc định của hình matplotlib bằng cách gán plt.rcParams['figure.figsize'].
- **Ý nghĩa & mục đích:** Hình mặc định thường nhỏ; khi xem heatmap chi tiết hay lưới nhiều ảnh cần phóng to để nhìn rõ. Đặt một lần áp cho các hình sau.
- **Code:**

```python
plt.rcParams['figure.figsize'] = (25, 25)
# hoặc
plt.rcParams['figure.figsize'] = [8, 6]
```
- **Visual:** Cùng nội dung nhưng khung to/nhỏ theo (rộng, cao) tính bằng inch.
- **Lưu ý:** Đây là cài đặt toàn cục — mọi hình sau đều đổi theo cho tới khi gán lại. Muốn chỉ đổi một hình thì dùng figsize=... trong plt.subplots/plt.figure thay vì rcParams.
- *Nguồn: visualize_activation, Week1Exercise3_NN_Backpropagation*

---

*Nguồn: 15 notebook trong `Docs/Lesson materials/`. Bản trích code+markdown (đã bỏ output ảnh) nằm ở scratchpad để tra cứu lại.*
