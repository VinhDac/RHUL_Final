# HANDOFF — tiếp tục dựng §4 (deep-plan) của dissertation "Data Snooping in Deep Learning"

*Dán vào đầu session mới. ĐỌC KỸ. Luật tối cao: NHẸ + TRỰC DIỆN + CÙNG-KHÁM-PHÁ + PLAN-TRƯỚC-EXECUTE-SAU.*

## 0. Đọc trước, đúng thứ tự
1. `memory/` tự load. Đặc biệt **`two-act-restructure.md`** = tài liệu MẸ của §4 (chi tiết mọi quyết định). Cũng: `writing-style-key-core`, `leveling-up-arc-3-5`, `working-mode-data-snooping`.
2. `NORTH_STAR.md` (la bàn) + `READ_ME.md` §0–§4 (§4 là bản đang dựng).
3. Handoff này (mục 2 = bài học đắt nhất — đọc kỹ để KHÔNG lặp sai lầm).

## 1. Cấu trúc chốt (2026-07-24)
- Dissertation = **HAI HỒI** trên **cùng bài toán thị trường**, không phải 4 dataset. Đã CẮT: Phone, §5 Lab, §6.
- **Hồi 1 (đau):** §2 Loan · §3 Market = cao trào (giấu kĩ năng DL, ngã từng trạm, xây khái niệm snooping từ thất bại).
- **Hồi 2 (làm chủ):** **§4** = giải lại bài thị trường như bậc thầy. Bài toán cụ thể: **định cỡ đệm rủi ro cho ngày mai (Value-at-Risk)** — model dự báo mai-động-hay-không (busy/calm) là ĐẦU VÀO cho cái đệm.
- **Hồi kết** (chưa làm): đặt tên **Snooping** + **workflow phổ quát** (chính là cái Plan §4).
- §4 = **ĐÚNG 3 PHẦN**: (a) Plan · (b) Execute · (c) Evaluate. Prose 3-phần hiện có trong READ_ME (khung "pricing") — SẼ ĐƯỢC VIẾT LẠI theo deep-plan (mục 3).

## 2. BÀI HỌC ĐẮT CỦA SESSION NÀY (đọc kỹ nhất)

### Sai lầm — ĐỪNG lặp:
1. **Dựng notebook/kết quả TRƯỚC khi có plan = KHÔNG TRUNG THỰC** ("xây rồi bịa kế hoạch để khoe"). User bắt đúng: cả §4 dạy plan-trước, nên CÁCH ta làm §4 phải plan-trước-execute-sau. → **PLAN xong mới EXECUTE mới EVALUATE.** Kết quả (số breaches...) chỉ thuộc Evaluate, sau khi tiêu chí đã cắm mù.
2. **Viết plan giọng TOÀN TRI ("ta đã lường hết") = BỊP người xem.** → Kể **KHÁCH QUAN, TỪ KINH NGHIỆM**: "sẹo §2/§3 dạy ta soi X, nên ta soi X", KHÔNG phải "ta biết X". Thêm **Tự-kiểm cuối** thừa nhận có thể vẫn sót giả định.
3. **Sa đà vào "lab" (cơ chế VaR/breach) → mất CORE.** Core của §4 = một **QUY TRÌNH nghĩ PHỔ QUÁT, chưng từ kinh nghiệm**; bài cushion chỉ là MỘT lần điền. Đừng biến §4 thành "một lời giải kỹ thuật khôn cho bài VaR".
4. **Khung "cái có ích" bị đảo nhiều lần** (sailor → pricing → VaR) vì tôi ĐOÁN. Chốt: "có ích" = **mục đích + hiệu chuẩn honest + chứng minh giá trị (breach coverage)**, KHÔNG phải accuracy. **KHÔNG jargon lạ** (VIX/quyền chọn/Black-Scholes làm loạn → đã bỏ; chỉ dùng "đệm/cushion", "thời tiết", lẽ thường).
5. **Nhồi 1 prompt quá nhiều → chất lượng TỆ** (user nói thẳng). → Làm **TỪNG MẨU, bàn từng mẩu, chất lượng > tốc độ.**
6. **Trang-plan 1-A4-visual = trống rỗng, thiếu lí do.** → Plan phải **SÂU** (nhiều technique, tại sao, cách cụ thể, contingency), không phải một thẻ gọn. (m4_plan.svg đã BỎ, không dùng.)
7. Vặt kỹ thuật: notebook `train_net` trả `(P,hist)` nhớ `[0]`; từng để sót đuôi mục cũ khi thay khối (luôn grep header sau khi replace-by-script).

### Đúng — GIỮ:
1. **Plan-trước honest order.** 2. **3 phần Plan/Execute/Evaluate.** 3. **Bài toán thật VaR/đệm** (research thật, số/fact thật); **bằng chứng = breach backtest**: cùng cỡ đệm, CỐ ĐỊNH 231 vỡ vs DỰ BÁO 200 (31 lần ít bị bắt hụt), dồn-cụm 24%→20%. 4. **Bản-án cắm TRƯỚC** kết quả (chống-snooping). 5. **Tự-kiểm cuối** (săn giả-định-còn-ẩn). 6. **Notebook = nguồn chân lý.** 7. Giọng plain, câu-neo đứng riêng, KHÔNG em-dash, ẩn dụ thuỷ thủ/thời tiết ở lớp trực giác.

## 3. DEEP-PLAN §4 — khung đang dựng (mục tiêu hiện tại)
Plan §4 = **flow lý-luận SÂU** (nặng TẠI SAO/purpose), khách-quan-từ-kinh-nghiệm, dựng **từng mẩu**:
- **0. Bài toán + KEY nó ép** (XONG — trong chat, chưa viết vào READ_ME): (a) target=độ lớn không hướng; (b) đầu ra=xác suất hiệu chuẩn; (c) "tốt"=honest không cao; (d) bền qua era + lỗi không dồn cụm; (e) dùng phòng thủ có biên. Mỗi cái nặng WHY. + bảng KEY→pipeline.
- **1–7. Đi từng bước pipeline**, mỗi bước theo khuôn: **nhắc NHIỀU technique → chọn + TẠI SAO → cách làm CỤ THỂ** (vd. split không chỉ "past→future" mà **walk-forward = expanding window/rolling origin/N fold**) → **mô phỏng thực thi: đúng thì X, sai thì SỬA cụ thể (đây là KEY của plan)**. Các bước: Frame · Split · Scale · Kiến-trúc · Huấn-luyện · Regularization · Measure. **BUNG FULL DL DEPTH**: activation (ReLU/tanh/sigmoid/leaky), số layer, độ rộng, epoch (cố định/early-stop), loss (CE/MSE/focal), optimizer (GD/momentum/Adam/RMSprop), GD (batch/mini/stochastic), regularization (L2/dropout/early-stop). Đầu ra thường vẫn là **lựa chọn ĐƠN GIẢN có lí do** (cú lật: khoe biết hết đồ nghề + phán đoán chọn cái đơn giản honest).
- **Đánh giá cuối:** lí thuyết (trần, hiệu chuẩn, winner's curse) + thực tế (breach), cân nhắc nhiều model.
- **Tự-kiểm cuối:** săn giả-định-CÒN-ẩn. Sắc nhất: **target nhị phân busy/calm là SIMPLIFICATION** của cái đệm thật cần (phân vị/độ lớn thực) — bê từ §3 chưa hỏi. Khác: data sạch không, tín hiệu còn ở chế-độ-mới không, HÌNH cái đuôi (black swan không có trong mẫu), feature đủ không. Kết: **KHÔNG chứng minh được là hết giả định ẩn** — "đây là chỗ ta VẪN chưa chắc", không phải "đã bắt hết".

## 4. Trạng thái file
- `READ_ME.md`: §0–§3 xong; §3f = recap+bridge; **§4 = 3 phần a/b/c bản "pricing"** (sẽ viết lại theo deep-plan). §6/§7 cũ còn (stale, thành Hồi kết). References tới §4 (thêm Kingma&Ba, Sutskever, Guo). Appendix A/B/C(market)/D(code).
- `notebooks/market_forecast.ipynb`: NGUỒN CHÂN LÝ §4, standalone, đã verify. Số: ceiling ~0.59–0.60 · arch 0.603/0.602/0.602 · opt GD/mom/Adam ~0.60 · search val~0.68 vs test 0.589 (là ERA không phải winner's curse, +0.004) · phân biệt 1.006% vs 0.532% (~2×) · ECE 0.014 (đã hiệu chuẩn) · **breach 231 vs 200 cùng đệm, dồn-cụm 24% vs 20%**. Hình m4_*: pipeline, ceiling, arch, convergence, search, discrimination, calibration, breach.
- `notebooks/market.ipynb` (§3), `loan.ipynb` (§2) standalone. Phone/lab.ipynb đã xoá. `figures/m4_plan.svg` BỎ (không dùng), `market_signal/market_tree.svg` thừa (dọn sau).

## 5. NHIỆM VỤ NGAY
Tiếp deep-plan §4 **từng mẩu, vừa-làm-vừa-bàn**: **Mẩu 1 — Frame** (nhiều cách đặt target: nhị phân trung vị / ngưỡng cố định / phân vị / hồi quy độ-lớn → chọn + tại sao → cụ thể → mô phỏng đúng/sai-sửa; thành thật nhị phân là simplification, nối Tự-kiểm). Rồi 2–7, Đánh giá, Tự-kiểm. Sau đó VIẾT LẠI prose §4 a/b/c theo deep-plan (giọng khách-quan-từ-kinh-nghiệm). Rồi: Hồi kết (đặt tên Snooping + workflow) · reconcile §1(dòng~65)/§2(dòng~92 "Section 5")/đánh-số(§4→§6 hụt §5)/Appendix D(thêm market_forecast.ipynb) · compliance (dẫn công thức base trong §1 + survey) · dọn SVG thừa · anonymity + Word-template.

## 6. Kỹ thuật
`py -3.14` (numpy/matplotlib/nbformat/nbconvert; KHÔNG dùng `.venv` hỏng; cairosvg KHÔNG có → SVG tay nhờ user liếc). Notebook: builder nbformat + ExecutePreprocessor(path=NBDIR), chart inline svg. Palette: accent #3a6ea5, muted #b8c0cc, đỏ #c85a52, grid #eef1f4. **Mode C**: đề 2–3 phương án, USER QUYẾT, mẩu nhỏ, KHÔNG commit git, LUÔN Read lại trước khi Edit (user hay sửa tay).
