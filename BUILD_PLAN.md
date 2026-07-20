# Build plan — hoàn thiện NỘI DUNG `READ_ME.md` (tầng thực thi)

*Tầng thực thi (execution layer). Sinh ra để **chống lạc lối khi làm lâu**. Luật làm việc: một mẩu nhỏ / lần, chốt xong mới sang mẩu sau, mỗi bước soi lại file này + `NORTH_STAR.md`. Đánh dấu tiến độ ở §5 sau mỗi lần chốt.*

**Thứ tự ưu tiên khi vênh nhau:** giọng & hồn = `NORTH_STAR.md` · chấm điểm = `Docs/Topic/Topic.png` + `handbook.txt` · nội dung từng chương = **file này** (refine `CHAPTER_MAP.md`) · lõi DL để trích/derive = `Docs/BAO_CAO_CORE_DL.md`.

**Phạm vi giai đoạn này:** CHỈ nội dung trong `READ_ME.md`. Doc (template Word) + HTML để cuối. Code hiện tại có thể đập đi, dựng lại thành **notebook-per-journey** ở giai đoạn sau; mỗi code phải **print ví dụ transform từng dòng**. **KHÔNG commit git.**

---

## 0. Câu cầu nối — LOGIC decision (✅ CHỐT 2026-07-19)

Toàn bộ arc phải phục vụ topic ("improve via architecture search + hyperparameter optimisation"), không được đánh nhau với nó.

> **CHỐT:** "Search chính là *cách ta cải thiện*; kỷ luật chống-snoop là *cách để con số ta giữ lại vẫn đáng tin* (báo cáo budget, niêm phong 1 test, không bao giờ chọn trên test). Ta không tin cái **Goal** (con số), ta tin cái **Process** tạo ra nó."

**Trạng thái:** đã chốt. Gieo ở cuối §1, thu hoạch ở §5/§6/§7.

---

## 1. Khuôn mỗi journey: VÒNG LẶP QUYẾT ĐỊNH *là* journey

**Động cơ (quan trọng nhất).** Một chương KHÔNG phải danh sách nhịp, mà là một người bình thường chạy đi chạy lại **một vòng lặp**, ở **mỗi nút** của cây quyết định, theo **nhịp suy nghĩ thật**. Không bao giờ đưa ra quyết định *đã rồi*; luôn cho thấy cái nhìn + lý lẽ dẫn tới nó.

**Vòng lặp (chạy ở mỗi nút):**
1. **Quan sát** — nhìn cái trước mặt (data thô / con số vừa ra). Thấy gì? Có gì gợn?
2. **Ra kế hoạch** — vậy thử gì, *vì sao*? Nói nước đi + kỳ vọng ra lời, **trước khi** làm.
3. **Làm** — chạy. Một con số.
4. **Đánh giá** — số so với kỳ vọng: xuôi không? *Cảm giác* là bản lề. Xuôi → đi sâu. Gợn → **gọi tên nghi (bold)**, quay lại ngay trước nước đi gây gợn, chọn nhánh khác → lặp.

**Chậm vs nhẹ (điều hoà, ĐỪNG lẫn lại — đây là lỗi đã mắc):**
- CHẬM = không nhảy cóc qua khúc cân nhắc; cho thấy đủ quan sát→kế hoạch→làm→đánh giá ở mọi nút, kể cả ngõ cụt. **Nhiều bước nhỏ**, không phải vài câu tóm tắt kết quả.
- NHẸ = mỗi bước một câu §0 ngắn, mộc. Không kịch, không dwelling.
- KHÔNG mâu thuẫn: §0 vừa chậm vừa nhẹ. Lỗi là (a) nhảy thẳng tới nước đã-quyết (quá nhanh), hoặc (b) prose nặng.
- **TEST một chương:** nếu một nước đi hiện ra mà độc giả chưa thấy ta *quyết* nó → đã bỏ một vòng lặp, quay lại cho thấy khúc quyết.
- **Giá:** chậm-mà-sống tốn chữ → câu càng phải ngắn, **figure gánh nhánh**, mắt để chừng 50 trang.

**Bộ xương 10 nhịp (arc phải có, SINH RA từ các vòng lặp trên):**

| # | Nhịp | Vai trò |
|---|---|---|
| 1 | Context | Data là gì, từ đâu — đọc bảng 3 dòng raw. |
| 2 | Muốn tìm gì | Việc thật, một câu. |
| 3 | Pipeline tại chỗ | Nhắc 5 bước cho case này (đừng thừa kế ngầm §1). |
| 4 | Quyết định + baseline | Quy tắc quyết định của case + chance. |
| 5–9 | **Thân journey** | Chạy vòng lặp ở mỗi nút: thử → đánh giá → **chất vấn (bold, nhịp 6)** → gọi tên ngõ cụt, vòng lại (7) → nhánh mới + control loại giả thuyết đối thủ (8) → thỏa câu hỏi (9). *Một chương thường chạy vòng lặp NHIỀU lần.* |
| 10 | Điểm cân bằng | Giữ số nào, bỏ bao nhiêu là leak/luck, có sửa quá tay không, chấp nhận thực tế tới đâu. |

**Nhịp 6 khác chất mỗi chương** (nếu không, độc giả thấy "một chiêu lặp 3 lần"): Loan=*đo sai thứ* · Market=*rò thời gian* + *thang trôi* · Phone=*rò qua người* (hỏi tầng định nghĩa: *"'unseen' nghĩa là gì khi người ở test đã có trong train?"*, đừng là "Market với người") · Lab=*may vs tài*.

**Cấu trúc chương + cây (đã chốt):**
- **Chia mục con a / b / c…** — mỗi mục là một khúc mắc / một nút cây, dễ theo dõi.
- **Show cốt lõi khi tiếp cận:** đi thẳng pipeline chuẩn (frame → split → scale → model §1 → train → đo), làm đúng sách, *kỳ vọng nó chạy*, rồi mới thấy vỡ. Dùng cốt lõi lộ ra, trỏ §1, KHÔNG derive lại. **Tôn trọng cốt lõi thì mới phản biện được cốt lõi.**
- **Cây lớn dần:** mỗi fork vẽ thêm một nhánh vào cây; **cuối chương: cây hoàn chỉnh + đoạn đánh giá** (cây đã dạy gì). Cây tổng ở Summit.
- **Trình tự mỗi mục** (đúng cách §1 giới thiệu DL): đọc đề → hình dung cách làm → làm theo cơ bản (tin nó giải được) → kỳ vọng kết quả → kết quả thật → hoài nghi (đúng/sai/cân bằng) → thử nhánh khác. Loop.

## 1b. Cách làm việc (LOCKED)

- **Chậm nhất, sâu nhất có thể.** Cùng trải hành trình: hoài nghi sự vật + mục đích, tìm câu hỏi chất vấn bản chất (đúng là gì, sai là gì, cân bằng là gì). Không vội.
- **Show TỐI THIỂU toán + code trong THÂN BÀI (directive locked).** Bài đi sâu vào **tư duy logic / bản chất**, KHÔNG trình diễn công thức. Thân chỉ nêu *cái logic vì sao* (+ hình gánh), **toán/code chi tiết → appendix**. F4 (derive): giữ phần **logic của derivation** trong thân (marker thấy mình hiểu), **full symbolic để appendix** — không mất trụ mà vẫn nhẹ.
- **Nội dung + notebook Jupyter xây SONG SONG.** Mỗi khúc: viết cell (code + **print transform từng dòng**) → chạy → lấy **số thật / figure thật** → rồi mới viết prose quanh số đó. Notebook = **nguồn-duy-nhất** cho mọi số + hình; prose không bịa số.
- Mỗi dataset = 1 notebook (loan/market/phone/lab) đồng hành chương của nó; appendix sau nhúng chính notebook đó.
- **Nhịp mỗi bước:** tôi đề 2-3 phương án logic → bạn quyết → làm cell + prose → soi NORTH_STAR + file này → chốt → tick tracker → bước sau.

---

## 2. Ba lỗ hổng LỚN phải đóng (high)

1. **Chiều sâu DL thật (ưu tiên 2 + topic F4).** §1 "What deep learning really is" đang chỉ có ẩn dụ ("mixed and bent", "little knobs", "nudges"), không nói net tính gì / loss là gì / gradient là gì. Công thức DL duy nhất bị đày ở Appendix B ("outside the page limit"). Toán derive lại là *xác suất* (Appendix A), không phải DL. → thêm 4 cơ chế DL vào thân bài + derive gradient DL cạnh Appendix A.
2. **Arc đánh nhau với topic (ưu tiên 2 + F2).** Search chỉ chạy trên noise, bị gán villain ("Search less"), chưa từng cho thấy cải thiện model thật. → thêm 1 beat **search-trung-thực-cải-thiện-model-thật** (budget + 1 test niêm phong). Chờ câu cầu nối §0.
3. **Beat 10 "điểm cân bằng" thiếu ở MỌI mục (ưu tiên 4, user nêu đích danh).** Kể cả §6/§7 chỉ nhị phân "đừng tin số / tin hiểu biết", không calibrate. Con số trung thực 0.633 / ~0.60 / 0.946 nằm sẵn để làm điểm cân bằng.

**Vừa (medium):** khúc giữa dễ thành "một chiêu 3 lần" (Phone cần câu hỏi khác chất) · Loan/Phone/Lab thẳng tuột, chỉ Market có fork-loop-back · §4 vứt beat learning-rate → nuôi thành hội tụ (E1) · §4 kết cụt · §1 câu hỏi thesis chưa bold.

**Nhỏ (low):** 1/3 cuối (§5-7) trôi sang manifesto; nhân-cách-hóa "the number lies/rots/betrays" chồng chất; §3 dòng 144 chèn citation giữa hành trình → đẩy xuống references.

---

## 3. Việc THÊM cho từng mục (checklist nội dung)

- **§1 DL intro:** +3-4 câu giọng §0 gọi tên forward / loss / gradient / step (cạnh hình "knobs"). **Bold** câu hỏi thesis ("how do we ever know it learned something real and not just the noise?") và để nó thở 1 nhịp. +đoạn derive gradient DL (chain rule qua softmax/CE + ReLU, `w ← w − η·grad`) cạnh Appendix A → **đóng F4**. Gieo câu cầu nối. Bỏ over-drama "the one that shadows this whole report".
- **§2 Loan:** biến 11-split thành nhánh-bị-tỉa thật (nêu split-theo-vị-trí 0.828 là nhánh *sai* đã bỏ). +1 dòng cân bằng: *giữ 0.633, không giữ 0.813, vì sao*.
- **§3 Market (mẫu chuẩn — giữ cấu trúc):** +1 dòng điểm cân bằng (giữ ~0.60 walk-forward, bỏ 0.03 là leak). Kéo câu suy nghĩ "One of the numbers is lying" lên bold/nổi. Đẩy citation Shimodaira/Gama xuống references.
- **§4 Phone:** +giả thuyết đối thủ + control (như Market); nuôi learning-rate 0.5 phân kỳ thành **beat hội tụ** (bước quá lớn vọt khỏi đáy, bước đúng lắng xuống) → chạm E1; +đoạn kết + dòng cân bằng (chance ~1/6 nên 0.946 vẫn mạnh); tách đoạn 7 câu dài (dòng 162) làm đôi.
- **§5 Lab (PHÁ BỎ + DỰNG LẠI SÂU HƠN):** bỏ noise-toy giả tạo; để winner's-curse/snooping hiện ra **từ chính cơ chế DL thật** — early stopping / seeds / epochs = tự-search (5 config lặng lẽ thành 150 tries). Phản biện cốt lõi *từ* cốt lõi. +beat search-trung-thực cải thiện model thật (F2); "search less" → điểm cân bằng. *Mở: giữ mỏ neo "sự thật đã biết" để đo gap thế nào — chốt khi dựng §5.*
- **§6 Valley:** thêm 2-3 dòng calibrate (mỗi case giữ bao nhiêu / bỏ bao nhiêu) trước khi sang §7; tách run-on dòng 221.
- **§7 Summit:** +1 ví dụ cụ thể "hiểu data → search trung thực → con số tin được"; nêu 0.633 / ~0.60 / 0.946 là các điểm cân bằng; mở rộng "understanding" gồm cả *net + optimiser chạy ra sao*, không chỉ *data*. Bớt giọng manifesto.

---

## 4. Thứ tự làm + kỷ luật

**Thứ tự đề xuất (đi từ nền ra):**
1. Chốt **câu cầu nối** (§0 file này) — LOGIC decision của user.
2. **§1 DL intro** — nền: 4 cơ chế DL + bold thesis + derive gradient + gieo cầu nối.
3. **§4 Phone** — pilot khuôn 10 nhịp (chương yếu flow nhất).
4. **§5 Lab** — beat search-trung-thực (đóng F2) + reframe "search less".
5. **§2 Loan** — nhánh-tỉa + điểm cân bằng.
6. **§3 Market** — thêm điểm cân bằng + dọn giọng (đã là mẫu chuẩn).
7. **§6 Valley + §7 Summit** — calibrate + ví dụ trust-earned + dọn manifesto.
8. Rà lại toàn bài: câu hỏi nhịp-6 có khác chất từng chương không; beat-10 có ở mọi chương không; giọng có nhẹ như §0 không.

**Kỷ luật mỗi bước (Mode C):** tôi đề 2-3 phương án cho mỗi lựa chọn logic → user quyết → tôi viết mẩu nhỏ → soi lại NORTH_STAR + file này → chốt → đánh dấu §5 → sang mẩu sau.

---

## 5. PROGRESS TRACKER (đánh dấu sau mỗi lần chốt)

- [x] §0 câu cầu nối — CHỐT
- [x] §1 DL intro — 4 cơ chế DL (forward/loss/gradient/step)  [bản A: gọi tên, không công thức trong thân]
- [x] §1 — bold câu hỏi thesis + để thở  [bản mộc: bold + xuống dòng, bỏ kịch]
- [x] §1 — gieo câu cầu nối  [bản mộc + câu constructive foreshadow "a number we can actually trust"]
- [x] §1 — F4 "ít toán": full symbolic → Appendix E (+ `backprop.svg`); thân §1 giữ đoạn 4-cơ-chế (prose) + `training_loop.svg` (mới) + 1 câu insight (p−y). §1 XONG.
      Còn nợ (khi dựng F2): derive Adam/optimiser-2 (cũng vào Appendix E).
- [x] §2 Loan — LÀM LẠI (a/b/c/d + `loan.ipynb` + `loan_tree.svg` + confusion.svg regen).  **§2 XONG.**
    - [x] a: reading the problem  [content §2 + loan.ipynb cell a: 30k, 22.1%, baseline 0.779, dòng hoán đổi được]
    - [x] b: by the book + split holds (0.817; position 0.83 pruned)  [content + loan.ipynb cell b]
    - [x] c: doubt (bold) → confusion (caught 459/1353, acc 0.819)  [content + loan.ipynb cell c + confusion.svg vẽ lại số thật]
    - [x] d: balanced accuracy 0.644 + điểm cân bằng (giữ 0.644) + `loan_tree.svg`  [content + loan.ipynb cell d]
- [x] §3 Market — LÀM LẠI (a/b/c/d + `market.ipynb` + `market_tree.svg` 2-trap regen; drift.svg giữ).  **§3 XONG.**
    - [x] a: reading the problem + trace transform TỪNG DÒNG  [content §3 + market.ipynb cell a: 6664 giá, index 677→7610, autocorr 0.287]
    - [x] b: the edge too easy (split leak: 0.618 vs 0.587, near-twins, direction control)  [content + market.ipynb cell b]
    - [x] c: killed by a unit (drift: 0.586→0.510→0.555; frozen |z| max 27.69)  [content + market.ipynb cell c + drift.svg giữ nguyên]
    - [ ] d: what the market cost us (balance + `market_tree.svg` regen 2-trap + carry §4)  ← ĐANG LÀM
- [x] §4 Phone — LÀM LẠI theo khuôn mới (a/b/c/d · cốt lõi lộ ra · cây) + `phone.ipynb` + `phone_tree.svg`.  **§4 XONG.**
    - [x] a: reading the problem  [content §4 + phone.ipynb cell a, số thật]
    - [x] b: first run + convergence scare (E1)  [content + phone.ipynb cell b: lr0.5→0.000 vs lr0.1→0.962]
    - [x] c: shuffle 0.966 → nghi (bold) → hold-out 0.948 (nhiễu) → KẾT Ở NGHI NGỜ  [content + phone.ipynb cell c]
    - [x] d: paired control (gap +0.022 mọi seed) + điểm cân bằng (giữ 0.947, khác Market) + `phone_tree.svg` + carry §5  [phone.ipynb cell d]
- [x] §5 "The search: the trap in us" — LÀM LẠI (a/b/c/d + `lab.ipynb`); T2 (feature loan + coin, truth 0.5) + **F2 giao đủ** (2 arch × 2 opt) + winners_curse.svg giữ.  **§5 XONG.** (không thêm cây riêng)
    - [x] a: the question + clean test (real features, coin labels, val-nhỏ/sealed-test)  [content §5 + lab.ipynb cell a; đổi title]
    - [x] b: early stopping snoops (best-val epoch 0.545 vs sealed test 0.500, gap +0.045)  [content + lab.ipynb cell b]
    - [x] c: seeds compound → data snooping (0.545→0.581→0.607; 4000 tries)  [content + lab.ipynb cell c + winners_curse.svg]
    - [x] d: F2 honest search (momentum 0.644→0.650, sealed) + cầu nối trọn + balance-point + carry §6  [lab.ipynb cell d: 2-hidden + momentum]
- [x] §6 Valley — recap 4 bẫy (số mới: 0.819/459 · 0.618/0.510 · 0.966/0.947 · 0.607/0.5) + câu hỏi khủng hoảng  [content]
- [x] §7 Summit — cầu nối chốt (không "đừng search"; ví dụ trust-earned 0.644→0.650 sealed; trust the Process kiếm ra số).  **MẠCH NỘI DUNG §0–§7 XONG.**
- [ ] Rà toàn bài (câu hỏi khác chất / beat-10 đủ / giọng nhẹ)
- [x] MỖI chương: notebook đồng hành (loan/market/phone/lab.ipynb) = nguồn số + figure  **XONG (§2–§5)**
- [x] DỌN cũ: xoá `code/*.py` (7 file); Appendix B→notebook pointer; Appendix D xoá; C/E trỏ notebook; §4→§5 refs; §5→App A link
- [ ] (sau) References → handbook style + **pass ẩn danh/metadata** (git author "Vin", metadata notebook/figure; kiểm không còn absolute path)
- [ ] (sau) đổ vào template Word (TemplateMScThesis.docx) → PDF
- [ ] (sau) HTML mind-map (đọc cùng figures/)

---

## 6. Ảnh chụp baseline (scorecard hiện tại, để đo tiến bộ)

| Mục | Câu hỏi | Sâu DL | Flow-tree | Bold | Giọng |
|---|---|---|---|---|---|
| §0 | mạnh | mạnh | mạnh | mạnh | nhẹ |
| §1 | ổn | **yếu** | mạnh | **yếu** | nhẹ |
| §2 Loan | mạnh | mạnh | ổn | mạnh | nhẹ |
| §3 Market | mạnh | mạnh | **mạnh★** | mạnh | trôi |
| §4 Phone | **yếu** | ổn | **yếu** | mạnh | nhẹ |
| §5 Lab | mạnh | ổn | ổn | mạnh | trôi |
| §6 Valley | mạnh | mạnh | ổn | mạnh | trôi |
| §7 Summit | ổn | ổn | ổn | ổn | trôi |

★ = mẫu chuẩn, dùng làm khuôn cho các chương khác.

*Đã đạt gần đủ: ưu tiên 1 (hành trình qua câu hỏi) + 5 (bold + giọng), 0 em-dash. Còn hụt: ưu tiên 2 (sâu DL + tôn trọng topic) + 4 (flow 10 nhịp + điểm cân bằng).*
