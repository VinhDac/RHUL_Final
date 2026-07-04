# Feedback — sửa Core.md cho sạch & đúng core mới

**Core mới (Hướng A).** Con số = `thật + may`; `may` = **bias do chọn lọc**, lớn dần theo mức tìm kiếm. **Deep learning là nơi tìm kiếm dữ dội nhất** — số epoch (early stopping = chọn lọc mỗi vòng), seed, width, lr đều thổi phồng *N hiệu dụng*, phần lớn **không được đếm** → winner's curse to & nguy hiểm nhất đúng ở DL. Ta **đo** gap (không tiên đoán bằng công thức), xác nhận trên loan/finance. **Niềm tin nằm ở quy trình (test niêm phong mở một lần, protocol phương sai thấp), không ở con số; `gap` = niềm tin mất trên mỗi đơn vị tìm kiếm.**

**Luật viết (ràng buộc).** Mọi công thức / hypothesis mở bằng **một ví dụ cụ thể dẫn dắt ý nghĩa trước** — người đọc hiểu trong một lần đọc. Không hoa mỹ (né lỗi "rhetoric > result"). **Nêu rõ giả định của mỗi công thức** — điều kiện nó đúng và chỗ nó sai (vd: `0.5/√n` chỉ đúng khi thước là *accuracy* — mỗi điểm đúng/sai — trên các điểm *độc lập*; sai với returns/Sharpe hay dữ liệu tự tương quan). **Không được assume người đọc hiểu ngay công thức bạn áp dụng.**

**Nguyên tắc kiến trúc (ràng buộc).** `Core.md` = **file mẹ** (thân bài): lập luận đơn giản, **đứng được một mình** — giám khảo có thể *không* đọc appendix/code. Nên: lý lẽ cốt lõi + ví dụ + con số then chốt nằm **trong thân bài**; **appendix** = đạo hàm đầy đủ (chiều sâu *tùy chọn*); **code** = *tái lập / kiểm chứng*, không phải chỗ để *hiểu*. Liên kết dọc: `claim → appendix → code → số`. Con trỏ code luôn *"reproduce/verify here"*, không bao giờ *"understand here"*. Lý lẽ thiết yếu **không** được chạy ra khỏi thân bài (giám khảo skip = coi như không có).

**Nối mạch (ràng buộc).** Mỗi phần/tiểu mục phải **MỞ bằng một câu nối với flow** — *"ta ở đây vì phần trước cần X"* — **trước** khi vào chi tiết/toán. Không liệt kê rời rạc; người đọc không bao giờ được mất mạch.

**Cách làm.** Đi từng §; tôi đề xuất ý → bạn chốt → mới viết vào Core.md. Giữ nguyên khung §1–§9.

---

## Kế hoạch nâng cấp (6 phase — phạm vi: README + Core.md + TẤT CẢ code + notebooks + appendix)

**Thước audit chung:** flow 5 nhịp `Nguyên nhân → Logic → Hành động → Kết quả → Đánh giá`, áp mọi phần.

**Tiêu chí PASS/FAIL (bắt buộc):** mô phỏng người đọc **trình độ THẤP NHẤT** (từ vựng ML tối thiểu, yếu ký hiệu, đọc tuyến tính). Một phần **ĐẠT** khi người đó — sau **MỘT lần đọc** — nói đúng được **ý đồ**. Không đạt → simplify.

**Chống decay (giữ hiệu suất qua nhiều phase):** `Feedback.md` = nguồn chân lý (không phải hội thoại) → đọc lại mục tương ứng trước mỗi phần · sub-agent/workflow **context sạch** cho audit/đọc file · checkpoint theo phase ra file · verify bằng đọc lại/chạy, không assert từ trí nhớ.

**Phases (0→1→2→3→4→5):**
- **0 — Audit nền (người mới, workflow): ✅ XONG → `AUDIT.md`** (6 chủ đề gốc · 9 phần trượt một-đọc · 13 ưu tiên).
- **0.5 — Foundation: ✅ XONG.** `config.py` = nguồn chân lý (synth 2000/200/**100k** · loan 2000/200/**10k** · finance 4000/200/**1k**; val=200 khắp nơi) · **freeze** data → `data/gspc_2026-07-03.csv` (lossless, `%.17g`) + `data/loan_uci350.csv` (offline, xác định, committable) · `models.py` cài **sklearn garden** (kNN/tree/logreg/SVM linear, thay stub) · dọn header pipeline.py + mlp.py "SS6"→§6 + README Layout (thêm mlp.py/config.py/data, sửa models.py row). Backend import sạch · test_lab xanh.
- **1 — Chạy 3 thí nghiệm (Mode C, print-check):** E-2 (§6.3) · Case 4 XOR (§5, §6.2) · H5 (§6.6). Chạy trước vì §5/§6 cần số thật.
- **2 — Viết lại Core.md §1–§7:** theo breakdown đã chốt, mở bằng nối mạch, flow 5 nhịp; Case 3 + backprop → appendix; thêm §6.3/Case 4/§6.6.
- **3 — Appendix + code + notebooks KHỚP:** đạo hàm đầy đủ + provenance; sửa models.py/finance split; thêm CI; mỗi "reproduce" trỏ cell thật.
- **4 — README:** hướng dẫn đọc + sơ đồ (đã có) + bố cục sửa lỗi thời + 1 dòng reproduce.
- **5 — Audit end-to-end (người mới, workflow, đối kháng):** README→Core→appendix→code; mọi link resolve · why-trước-what · core↔code liền mạch · chỗ nào vẫn bí/nghi/thiếu bằng chứng.

**Thứ tự thực thi (INTERLEAVE — chốt 3 Jul):** viết **§1→§4 trước** (không cần số) → rồi §5→§7 kèm số. ⚠️ Background compute flaky ở env này (nuốt output ×2) → số §5+ chạy **foreground-lean per-section** (R vừa timeout 600s + noise band) cho chắc, thử background khi được. Nội dung/nguyên tắc không đổi, chỉ đổi thứ tự thực thi.

**Phase 1 — kết quả (checkpoint, đang chạy):**
- **§5 Headline ✅ (Case 1, n_test=100k, R=8):** apparent 0.51→0.58, true ~0.50, **gap +0.009→+0.080** (N 1→200), gap_std ±0.008–0.03.
- **§5 Optimal budget ✅ (Case 2+20%nhiễu):** apparent 0.70→0.80; true đỉnh ~0.766@N50 → 0.760@N200 (**dip nhẹ, trong nhiễu** → khung honest); gap +.006→+.040.
- **Isometry ✅ (→appendix, d=20):** kNN 0.805=0.805 · logreg 0.989=0.989 · SVM 0.986=0.986 · **tree 1.00→0.71 (−0.29)**.
- ⚠️ **Figures cần REGEN (Phase 3)** với số canonical: headline_gap_vs_N · optimal_budget · isometry.
- **Case 4 / §5 MLP-bắt-buộc ✅:** XOR — logreg 0.49 · SVM 0.49 (chance) vs **MLP ~0.99** (width 16–256, robust). Garden: kNN 0.88, tree 0.95 (phi tuyến cũng học → khung "vs baseline tuyến tính").
- **§6.6 H5 (thuốc) ✅ — PASS (Case2+20%noise, R=10):** A(tham 20cfg×3seed×6ep) apparent **0.805** / true **0.753** / gap **+0.052** vs B(honest 10cfg+5fold) apparent 0.762 / true **0.765** / gap **−0.003**. → B thật ≥ A (+0.012) VÀ số B tin được (gap≈0). *"A đẹp mã hơn, tệ hơn."* Tiêu chí: tin số khi vượt baseline > `0.5/√n` trên test niêm phong trung thực.
- **§5 loan/finance ✅:** loan logreg 0.821>0.787 (signal), gap +0.006→+0.035, stakes 58% nợ xấu duyệt nhầm (acc 0.815). finance logreg 0.540≈chance, edge 0.57→0.53 (luck), gap +0.007→+0.038.
- **§6.3 E-2 (Hd) ✅ trụ DL — PASS (Case 1, 5cfg×5seed×6epoch, R=20):** gap tăng theo chiều ẩn — config-only(N_eff=5) **+0.033** · +seed(25) **+0.055** · +epoch(150) **+0.072**. Mỗi mức **rơi đúng đường headline** tại N_eff (5→.031, 25→.057, 150→.077) → *"nút ẩn = N trá hình"*; 5 config tưởng-là gánh gap ~150. (gap3 hơi dưới headline vì draws trong 1 config tương quan → effective-N <150, honest.)
- **Case 4 / §6.2 H3 tái kiểm ✅ (canonical R=10, +30% nhiễu):** gap-vs-width — **Case 2:** .028/.026/.022/.029/.009/.040/.029 · **Case 4 (XOR):** .027/.019/.012/.030/.024/.047/.034 (width 4→256). Cả hai **phẳng ~0.01–0.05, KHÔNG xu hướng theo width** (scatter trong ±0.03 nhiễu). → **H3 bị bác NGAY CẢ trên phi tuyến** — capacity không làm gap to *kể cả nơi cần capacity để fit*. Nút nguy = **N + nhiễu val**, không phải size. *(Compute: sweep nặng vượt timeout foreground 600s → heavy runs sau chạy background + verify file.)*

---

## Flow bài (đã sửa) + Biện pháp mới (H5)

**Spine:** Vấn đề → dụng cụ đo → đo trong lab → cái gì chi phối gap → nó cắn ngoài đời → **thuốc + tiêu chí** → kết.

1. **Vấn đề** (§1) — điểm = thật + may; DL tệ nhất (nút ẩn epoch/seed).
2. **Dụng cụ đo** (§2–3) — lab tự sinh + cái máy → gap.
3. **Đo trong lab** (§4–5) — N↑ → gap↑ (headline).
4. **Vặn từng nút** (§6.1–6.4 + Hd) — nhiễu→to; model to→không; nút ẩn DL→to; protocol trung thực→nhỏ.
5. **Đời thật** (§6.5) — loan dịu; finance edge 64%→53%, mất tiền.
6. **⭐ BIỆN PHÁP (MỚI — §6.6, H5)** — so sánh **A (snoop)** vs **B (trung thực)** → B thật ≥ A thật, gap nhỏ → tin được. **Tiêu chí chấp nhận:** vượt baseline hơn cả sai số `~0.5/√n`, trên test trung thực.
7. **Kết** (§7) — niềm tin ở quy trình, không ở con số; gap = niềm tin mất/đơn vị search; DL nguy nhất + đã có thuốc.

**Bỏ:** hypothesis "simple > complex" theo nghĩa *kích thước model* (mâu thuẫn hướng DL phi tuyến + đã bị H3 bác). Giữ theo nghĩa **quy trình** (trung thực-kiềm chế > tham-mù) = chính H5.

**Kéo theo:** thêm mục **§6.6** (thí nghiệm A vs B) — cần data biết truth (synthetic, hoặc test lớn trung thực); H5 dùng lại đúng cái máy §3.

---

## §1 Introduction — ĐÃ CHỐT (chưa viết vào Core.md)

Giữ nguyên 4 nhịp, thay ruột:

- **[TRỤ] #3 — nhịp "Why it matters" → "Vì sao DL tệ nhất":** cắm luận điểm nút ẩn — *epoch* = chọn lọc mỗi vòng, *seed* = một lượt bốc → N **hiệu dụng** lớn hơn nhiều số config ta tưởng. Đây là chỗ giành nhãn DL. *(bắt buộc)*
- **[TRỤ] #4 — nhịp "Central hypothesis" → "What we claim":** hạ *"gap grows with N"* xuống **sự thật nền tảng**; luận điểm trung tâm mới = **niềm tin ở quy trình**, `gap` = niềm tin mất / đơn vị tìm kiếm. Viết câu "trust" **mộc**, không hoa mỹ. *(bắt buộc)*
- **#2 — nhịp "Hidden flaw":** dẫn bằng **ví dụ cụ thể trước** (đứa may nhất trong 10 học sinh dốt được 80%) rồi mới tới `true + luck`. **Chỉ MỘT dòng gieo** ở §1 — để §3 mới khai triển analogy (tránh lặp ba lần).
- **BỎ #1:** KHÔNG thêm epoch/seed vào câu mở đầu. Giữ câu mở ở nút *có ý thức search* (architecture / width / lr) — để dành epoch/seed cho **cú lật** của #3 (đếm vs không đếm).
- *(tùy chọn, ưu tiên thấp)* nửa dòng cuối beat 4 hé lộ real-data (loan/finance), hoặc để §2.4 lo.

**Breakdown tối giản (4 câu · thứ tự KHÓA bởi logic 1→2→3→4):**
1. **Thói quen** — thử nhiều cấu hình → giữ val cao nhất → gọi là "cải tiến". *(nền, ~1 câu)*
2. **Cái bẫy** — val = thật + may; giữ best-of-N = giữ đứa may → số phồng. *(vd: 10 người đoán mò, đứa may nhất 80%)* *(nền, ~1 câu)*
3. **DL nặng nhất** — epoch + seed là nút ẩn → N thật lớn hơn nhiều → phồng to nhất. *(TRỤ — dồn chữ)*
4. **Ta làm** — đo `gap = val − test` (mở 1 lần); tin ở quy trình, không ở con số. *(TRỤ — punchline)*

Thứ tự khóa: 2 cần 1, 3 cần 2, 4 phải cuối. Phân bổ chữ: 1,2 ngắn nhất — dồn cho 3,4.

**Cam kết kéo theo:** chốt #3 = phải chạy **E-2** (đo gap khi coi epoch/seed là chiều search) để câu "gap grows under hidden DL search" có bằng chứng.

---

## §2 Background — ĐÃ CHỐT (chưa viết vào Core.md)

**Breakdown (thứ tự KHÓA 2.1→2.2→2.3→2.4):**
1. **2.1 Ba tập** — train (dạy) · val (chọn) · test (chấm thật, niêm phong). Ta khoe val, cái cần là test. *(nền)*
2. **2.2 Cơ chế + DL** — max của N điểm nhiễu → đứa may → gap. **+ N thật = configs × seeds × epochs** (nút ẩn; cầu §1 → §6/E-2). *(TRỤ — dồn chữ)*
3. **2.3 Vì sao lab** — data thật không biết truth → đo không sạch; tự sinh thì biết. *(nền)*
4. **2.4 Roadmap** — synthetic → loan → finance **→ + thuốc (§6.6)**. Cung = **đo → xác nhận → chữa**. *(hé lộ, 1 câu)*

**Sửa kèm:**
- **[P0] 2.2 — mean-zero có điều kiện:** `E[Ŝ_i | model_i] = S_i`, cần val i.i.d.; **vỡ trên finance** (cửa sổ tự tương quan).
- **2.2 —** ví dụ trước, ký hiệu sau; không kể lại analogy thi cử (trỏ §1); bỏ câu lặp *"three names, one phenomenon"*.
- **2.3 —** nói mộc lại câu *"ruler carries the same disease"*. *(ưu tiên thấp)*
- **Toán:** intuition + ví dụ trong thân bài; đạo hàm winner's curse đầy đủ → **appendix**. Chưa cần con trỏ code (để §5/§6).

---

## §3 Method — ĐÃ CHỐT (chưa viết vào Core.md)

**Breakdown (thứ tự):**
1. **Analogy** — thi cử: N thí sinh · quiz ngắn (val) · thi khổng lồ (test). *(mở, dễ hiểu — cắt còn 1 hình)*
2. **Cái máy 6 bước** — chia → search N → train → chấm val → giữ best → **mở test 1 lần** → gap. *(lõi)*
3. **Công cụ = MLP** — search width, lr **+ epoch, seed (nút ẩn)**. *(TRỤ 1 — DL; = E-2)*
4. **Early stopping = chọn lọc** — mỗi epoch một ứng viên → §6.4 giải thích vòng lặp → epoch là nút chọn. *(TRỤ 2 — cứu §6.4)*
5. **Cây thước = test niêm phong** — `0.5/√n` → sự thật sắc. *(nền)*
6. **Kỷ luật** — mở một lần + mỗi thí nghiệm vặn một nút. *(nền)*

**Nặng nhất:** Trụ 1 (epoch/seed) + Trụ 2 (early-stopping = chọn lọc, cứu §6.4). Còn lại ngắn.

**Toán & kiến trúc (đã chốt):**
- `gap`, `argmax`, `N_eff`: mỗi cái 1 dòng.
- `0.5/√n`: **intuition (đồng xu) + số + giả định** (accuracy, điểm *độc lập* → **không áp finance**) ở **thân bài**; **full derivation → appendix**.
- Backprop / §6.4 đạo hàm → **appendix**.
- **KHÔNG link code, KHÔNG giải thích code** trong §3 (thân bài sạch; code = bằng chứng ở Appendix A).

**Cam kết:** Trụ 1 = phải chạy **E-2**.

---

## §4 Synthetic lab — ĐÃ CHỐT (chưa viết vào Core.md)

**Mở bằng mối nối:** §3 cần *biết sự thật* để cây thước sắc → §4 là chỗ **tự tạo dữ liệu có sự thật biết trước**; "đặt nhãn" = "đặt sự thật".

**Các case = chuỗi ÉP BUỘC (không liệt kê):**
- **Case 1** (nhãn ngẫu nhiên, truth = 50%) — ép bởi *"cô lập may"*.
- **Case 2** (`sign(x₁)`, tuyến tính) — ép bởi *"cho thấy cái giá"* (over-search → model thật tệ đi; nền cho H2/H3).
- **Case 4** (`sign(x₁·x₂)`, XOR phi tuyến) — ép bởi *"làm MLP bắt buộc + tái kiểm H3"* (plan dưới).
- **Case 3** (isometry / xoay) — **XUỐNG APPENDIX** (đẹp nhưng lạc mạch snoop; E-2 đã gánh nhãn DL). Dùng sklearn garden.

**Toán & hình:** đại số isometry → appendix. **1 hình 2D** (`x₁` vs `x₂`: loạn / dọc / bàn cờ) → thân bài; code sinh hình → notebook. Hình = **support cho logic rõ, KHÔNG phải trọng tâm** (1 hình, tối giản).

### Case 4 (phi tuyến) — ĐÃ CHỐT: XOR
**Vì sao:** E-2 làm snoop tệ nhất ở DL, nhưng chưa cho thấy MLP *làm được điều model đơn giản không làm được*. Case 4 = quy luật phi tuyến, đường thẳng thua, chỉ MLP học.
**Hai việc:** (1) MLP **bắt buộc** (DL-as-model, bổ sung E-2 = DL-as-worst-habitat); (2) **tái kiểm H3** ở nơi capacity thật sự cắn.
**Thiết kế:** nhãn `y = sign(x₁·x₂)` (XOR — **đã chốt**) — thẳng = 50%, cân bằng 50/50, MLP học được.
**Đo:** (1) logreg/SVM ≈ 50% vs MLP ≈ 100% (bảng + hình bàn cờ); (2) quét width trên Case 4 + nhiễu → gap có tăng theo capacity không (phẳng = bác H3 mạnh hơn; tăng = phát hiện DL mới).
**Vị trí:** định nghĩa §4 · "MLP bắt buộc" §5 · H3 tái kiểm §6.2. **Cam kết:** phải chạy 2 thí nghiệm.
**KHUNG ĐÚNG (Phase 0.5 xác nhận bằng số):** "MLP vs baseline **TUYẾN TÍNH** (logreg/SVM ≈ 50% trên XOR)" — **KHÔNG** phải "chỉ MLP", vì kNN/tree cũng học XOR (0.89/0.99). Điểm: MLP học ranh giới mà model *tuyến tính* **chứng minh không thể**.

---

## §5 Core results — ĐÃ CHỐT (chưa viết vào Core.md)

**Mở bằng mối nối:** §4 dựng lab *biết sự thật* → §5 **chạy máy, đọc gap**. Lần đầu bài có con số → nơi con trỏ *"reproduce"* bắt đầu.

**Breakdown (nặng → nhẹ):**
1. **Headline** (Case 1) — N↑: apparent leo, true phẳng 0.5 → gap `0 → +0.087`. *(TRỤ — hình trung tâm)*
2. **Optimal budget** (Case 2 + nhiễu) — true đạt đỉnh rồi **TỤT** → over-search mua model tệ hơn. *(TRỤ — cái giá)*
3. **MLP bắt buộc** (Case 4) — logreg ≈ 50% vs MLP ≈ 100%. *(mới — DL as model)*
4. **Fit ≠ generalise** (Case 1) — fit train 100% mà test 50%. *(hệ quả, 1 câu)*
- *(Isometry → appendix.)*

**Vai (đã chốt):** §5 = SHOW. Mỗi kết quả = **hình + số (đo) + 1 câu nối về §2.2 + con trỏ reproduce**. KHÔNG đạo hàm (measure, not predict), KHÔNG giải thích code, KHÔNG lặp "why". Magnitude là *đo* không suy (tối đa 1 câu *bằng lời* "vì sao bão hòa" — config tương quan; không công thức).
**P0:** bảng số phải có **thanh sai số / CI** (`gap_std` đã tính) — cả bài nói về nhiễu mà bảng không có sai số là mỉa mai.

---

## §6 Extensions + đời thật + thuốc — MAP ĐÃ CHỐT (đang khoan từng tiểu mục)

**Mở bằng mối nối:** §5 đo gap trong lab → §6 **vặn từng nút** (chi phối gap) → **đời thật** → **thuốc**.

| Tiểu mục | Việc | Trạng thái |
|---|---|---|
| **6.1 Nhiễu (H2)** | nhiễu↑ → gap↑ | giữ |
| **6.2 Capacity (H3)** | model to → gap không đổi (bác) **+ tái kiểm Case 4** | giữ + Case 4 |
| **6.3 Nút ẩn (Hd=E-2)** | epoch/seed → N thật↑ → gap↑ | **MỚI — trụ DL** |
| **6.4 Protocol (H4)** | k-fold → gap↓ (thuốc đầu) | giữ |
| **6.5 Đời thật** | loan dịu + finance mất tiền | giữ **+ sửa P0** |
| **6.6 Biện pháp (H5)** | A vs B → tin được + tiêu chí chấp nhận | **MỚI — thuốc** |
| ~~backprop math~~ | đạo hàm forward/backprop/SGD | **→ APPENDIX** |

Cụm: chi phối gap (6.1–6.4) → đời thật (6.5) → thuốc (6.6). **Nặng: 6.3, 6.5, 6.6.**
**P0 gom ở 6.5:** finance overclaim (long-short vs long-only + phí); `gap = curse + sai-số-test`; "universal" nhẹ tay với n=2.

### §6.1 Nhiễu nhãn (H2) — ĐÃ CHỐT
- **Mở nối:** §5 đo gap → §6 vặn từng nút; nút đầu = nhiễu.
- **Cơ chế (1 câu):** sạch → accuracy đụng trần → best-of-N không chỗ vọt → gap≈0; nhiễu rời trần → có chỗ → gap↑.
- **Kết quả:** gap `0 → +0.062` (flip 0→0.5), đơn điệu *(1 hình)*.
- **Nối:** xác nhận **H2**; gap = "chỗ trống cho may" (nhiễu + N), **không phải model to** → gài sẵn H3. 6.1 & 6.2 là một cặp.
- **Vai:** theo luật §5 (show + 1 câu cơ chế + hình + reproduce). Không đạo hàm, không giải thích code.

### §6.2 Capacity (H3) + tái kiểm Case 4 — ĐÃ CHỐT
- **Mở nối:** §6.1 gap = "chỗ trống cho may" → hỏi: model TO có làm gap to? (trực giác: net to→overfit→gap to).
- **Case 2 (tuyến tính):** **BÁC** — gap phẳng width 4–256. Cơ chế: gap = độ vọt max-of-N (N + nhiễu val), size không nằm trong.
- **Thành thật:** trên tuyến tính net nhỏ đã đủ → "phẳng" gần như **bị ép** → cần test công bằng.
- **Tái kiểm Case 4 (XOR):** capacity thật sự cần → **phẳng = bác mạnh hơn; tăng = phát hiện DL mới**. Báo trung thực.
- **Mấu chốt:** Case 4 biến H3 từ *null-gần-như-bị-ép* → **phép thử thật**.
- **Vai:** luật §5 (show + 1 câu cơ chế + 2 hình + reproduce). Không đạo hàm.

### §6.3 Nút ẩn (Hd = E-2) — ĐÃ CHỐT ⭐ trụ DL (make-or-break)
- **Mở nối:** 6.1+6.2 → gap do N + nhiễu val. Nhưng "N" tới giờ = config *cố ý*. Cú lật: chiều N ta KHÔNG đếm.
- **Hd:** epoch (early stopping) + seed → N hiệu dụng↑ → gap to hơn nhiều số config tưởng.
- **Ví dụ:** `5 config × 3 seed × 300 epoch = 4.500` ứng viên.
- **Thiết kế:** coi epoch/seed là chiều search; đo gap khi N thật lớn dần; so *đếm-config* vs *đếm-config×seed×epoch*.
- **Vì sao là DL:** logreg ~1 nút; DL bùng nổ nút ẩn → curse tệ nhất ở DL → **giành nhãn "deep learning"**.
- **Tinh tế:** early stopping tốt cho *training* nhưng lấy val-score epoch đẹp làm số báo cáo = **snoop**.
- **Trạng thái:** **PHẢI CHẠY (E-2)**, make-or-break; số *đo* không suy.
- **Vai:** luật §5 (show + ví dụ + hình + reproduce). N_eff = phép nhân, không đạo hàm.

### §6.4 Protocol (H4) — ĐÃ CHỐT (thuốc đầu)
- **Mở nối:** 6.1–6.3 = cái gì làm gap TO → giờ: cái gì làm NHỎ? → thuốc đầu = chọn model trung thực hơn.
- **H4:** protocol phương sai thấp (k-fold) → gap nhỏ.
- **Cơ chế:** val nhỏ một mảnh = nhiễu → best-of-N vọt xa; k-fold trung bình k lần = ít nhiễu → vọt ít → gap ~nửa.
- **Kết quả:** ~giảm nửa (N=50: single `+0.074` vs 5-fold `+0.026`) *(1 hình)*.
- **Nối:** thuốc ĐẦU TIÊN → dẫn §6.6 (H5).
- **Thành thật (P1):** k-fold apparent = `best_cv` vs single = `best_val` → **không cùng thước**; nêu rõ hoặc chỉnh.
- **Vai:** luật §5; "giảm phương sai" nói bằng lời, không công thức.

### §6.5 Đời thật (loan + finance) — ĐÃ CHỐT (external validity + P0)
- **Mở nối:** 6.1–6.4 đo gap trong lab (biết truth); lab không chứng minh cắn thật → external validity → cùng máy trên real data.
- **Loan (có tín hiệu):** gap dịu nhưng có (+0.010). Stakes: 82% acc mà duyệt sai **66% nợ xấu**.
- **Finance (tín hiệu≈0):** edge **64% → 53%** out-of-sample.
- **P0 SỬA:**
  - **Finance "mất tiền" overclaim:** +82 vs +94 phần lớn *long-short vs long-only + không phí*. Snoop point (64→53) đúng; sửa khung → trừ phí + so cùng loại, hoặc nhấn snoop.
  - **`gap = curse + sai-số-test`** (không phải đẳng thức): finance n_test~1000, SE~0.016 ~ gap → nêu rõ.
  - **"universal" nhẹ tay:** gap real (+0.010, +0.019) có thể trong nhiễu của 0; n=2 → caveat/CI.
  - **Finance split cố định** (rng bỏ) → 1 realization → không thanh sai số → nêu.
- **Nối kiến trúc:** `0.5/√n` KHÔNG áp finance (tự tương quan) → **lý do lab tồn tại**.
- **Reproduce** + sửa: split 1500/50/1000 vs 4000/200/1000 thống nhất; models.py stub.
- **Vai:** luật §5 **+ trung thực nặng** (limits).

### §6.6 Biện pháp (H5) — ĐÃ CHỐT (thuốc đầy đủ, PHẢI CHẠY)
- **Mở nối:** §6.5 = curse cắn thật; cả bài tới giờ CHẨN bệnh, §6.4 hé lộ thuốc → §6.6 = thuốc ĐẦY ĐỦ: **A vs B trực tiếp + tiêu chí chấp nhận**.
- **Thiết kế:** A (tham) = search khổng lồ (N to + epoch/seed ẩn), val nhỏ, khoe best-val. B (trung thực) = search nhỏ, k-fold, N có trần. **Mở test niêm phong cho cả hai.**
- **Kỳ vọng:** **B thật ≥ A thật, gap B nhỏ hơn nhiều → B tin được.**
- **Tiêu chí chấp nhận (usable):** chấp nhận model KHI điểm test *trung thực* vượt baseline (đoán mò/incumbent) **hơn sai số `~0.5/√n`**. KHÔNG phải con số cố định (70%).
- **Đo ở đâu:** cần biết truth → lab lý tưởng (B_true ≥ A_true chính xác); có thể kèm loan.
- **Vì sao "thực tế":** biến bài từ *chẩn bệnh* → *chẩn + kê thuốc + tiêu chí dùng được sáng thứ Hai*.
- **Trạng thái:** PHẢI CHẠY (H5), make-or-break cho phần remedy.
- **Vai:** luật §5; giao ra **tiêu chí chấp nhận** (bằng lời, chỉ dùng `0.5/√n` đã có).

### §7 Kết luận — ĐÃ CHỐT (DISCUSSION: synthesis, KHÔNG toán/code/số mới)
- **Mở nối:** §6 = chẩn + đời thật + thuốc → §7 gom thành kết luận.
- **Mở bằng nghịch lý:** "biết test xấu thì sửa thế nào để **không tự lừa**?" → dẫn tới câu trung tâm.
- **Kết luận trung tâm:** niềm tin ở **QUY TRÌNH**, không ở con số; `gap` = niềm tin mất / đơn vị search.
- **Gom 4 mảnh** (mỗi 1 câu, trỏ §): cơ chế đo được (gap↑N) · DL tệ nhất (nút ẩn) · cắn thật (loan/finance) · thuốc (H5 + tiêu chí).
- **Internal + external validity:** kết luận **LÀ phép so sánh**, không phải một dataset.
- **Honest limits:** thước cùn trên real data (finance 1 realization); **E-2 / Case 4 / H5 là cam kết mới PHẢI chạy** — chưa có số thì không kết như đã có.
- **Vai:** DISCUSSION — không toán/code/số mới; chỉ nối về core statement.

---

## Track A — tiến độ viết Core.md
- **§1–§7 ✅ ĐÃ GHI CORE.MD (TOÀN MẠCH XONG)** — mỗi § mở bằng nối mạch, ví-dụ-trước, gloss jargon, số canonical (frozen data), P0 finance đã sửa (honest), thêm §6.3 Hd/E-2 + §6.6 H5. Còn §0 abstract / §8 / §9 (write-last).
- **Phase 3 chunk 1 ✅:** backprop→Appendix B · isometry→Appendix C · roadmap cập nhật · 0 scaffold ✅ sót · token Việt/Phần dọn · §6 renumber sạch.
- **Phase 3 chunk 2 ✅:** `figures/make_figures.py` regen **8 figure canonical** (headline/optimal/noise/capacity+Case4/protocol/isometry/cases_2d, giữ exam); xóa 4 stale; mọi ref resolve.
- **Phase 3 chunk 3 ✅:** `experiments.py` (formalize E-2 `hidden_n_levels` + H5 `remedy_AB`); 3 notebook (01 core/AppC · 02 real · 03 extensions) rebuilt sạch, gọi backend + config, deterministic seed=0 → chạy reproduce Core.md. Mọi code cell compile + fast cells verify OK. *(Notebook chưa embed output — reader chạy; sweep vài phút/cái.)*
- **Phase 3 chunk 4 ✅:** Appendix A (import root + module map + numbers-note + Case 4 row, finance 0.544→0.543); tests `lab/mlp/pipeline` → **`sys.exit(1)` khi FAIL** + coverage run_once/gap_kfold + Case-4, cả 3 PASS. **→ PHASE 3 XONG.**
- **Phase 4 (README) ✅:** primer (configuration + analogy + gap-direction), Layout + experiments/data + fail-loud tests, Run offline/frozen/seed=0.
- **§0 abstract ✅ · §9 how-to-use ✅ · §8 self-assessment khung ✅** (phần cá nhân để user điền).
- **Phase 5 (audit cuối) ✅:** workflow 5-auditor — mọi §1–§7 PASS one-read; punch-list nhỏ đã đóng (roadmap §1 aims, backprop ref §6.4→App B, Case-4 label XOR đúng, finance 6654→6658, gap_kfold section 6.3→6.4). Cross-ref nội bộ resolve; P0 Phase-0 đóng hết.
- **🎯 6 PHASE HOÀN TẤT** — Core.md §0–§9 + App A/B/C sạch một-lần-đọc, số canonical trên frozen data, notebooks reproduce, tests fail-loud, README + figures đồng bộ. Chỉ còn §8 phần cá nhân (user điền) + commit.
- **[Phase 3 — đừng rơi] Chuyển nội dung isometry (từ §4 cũ) → appendix** (đại số orthogonal + kNN/logreg/SVM bất biến, tree tụt); §4 body chỉ trỏ.
- **[Phase 3 — đừng rơi] Sinh `figures/cases_2d.svg`** (3 panel 2D: random/dọc/bàn cờ) từ notebook; §4 đang trỏ.
- **Đang làm:** §5 (cần số → chạy foreground-lean).

## Backlog P0 (bàn khi tới § tương ứng, đừng để rơi)
- Finance "mất tiền" overclaim (long-short vs long-only + phí giao dịch) — §6.5
- `gap = curse + sai-số-test` (không phải đẳng thức trên real data) — §2.2/§3/§6.5
- Tái lập: `models.py` stub, finance split 1500/50/1000 vs 4000/200/1000 — §5/§6.5/Appendix
- Thiếu thanh sai số/CI ở các bảng (dù `gap_std` đã tính) — §5/§6
- "universal" quá mạnh cho n=2 real dataset — §7
