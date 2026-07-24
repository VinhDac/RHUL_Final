# HANDOFF — tiếp tục dựng dissertation "Data Snooping in Deep Learning"

*Dán vào đầu session mới. Chắt lọc từ session dựng lại trọn §3 Market. ĐỌC KỸ trước khi làm gì. Luật tối cao: NHẸ + TRỰC DIỆN + CÙNG-KHÁM-PHÁ.*

Bạn (Claude) tiếp quản MSc dissertation "CS: Deep Learning" viết kiểu HÀNH TRÌNH: một người bình thường đi qua pipeline DL, mỗi dataset là một journey lộ ra một **hidden assumption**. §0–§2 là bản mẫu. Session này đã **dựng lại trọn §3 Market (a→e)** theo giọng mới. Nhiệm vụ tiếp: **§3f**, rồi §4 Phone, §5 Lab, §6, §7.

## 0. Đọc trước, theo thứ tự
1. memory/ tự load — đặc biệt `leveling-up-arc-3-5` (mọi thiết bị của §3) và `writing-style-key-core` (luật giọng).
2. `NORTH_STAR.md` (la bàn) + `LOAN_PLAN.md` (giọng đã khoá).
3. `READ_ME.md` §0–§3 (§3 là bản mẫu MỚI nhất — bắt chước giọng này, KHÔNG bắt chước §4–§7 cũ).
4. Handoff này (mục 2 = phần quý nhất).

## 1. Trạng thái
- ✅ **§3 Market a→e DỰNG XONG** (giọng plain/exam, dẫn cảm xúc). §3f (hạ cánh) **VẪN BẢN CŨ** → việc kế tiếp: viết lại §3f cho khớp giọng + gắn câu chốt user đã thêm.
- ✅ Notebook `market.ipynb` rebuilt (config B), 13 hình §3 trong `figures/`.
- ⏳ **§4 Phone, §5 Lab, §6, §7 VẪN BẢN CŨ** (config + giọng cũ) → dựng lại sau.
- ⚠️ Nợ: **§6 Valley trích SỐ MARKET CŨ** (0.618…) → reconcile sang 0.612/0.587/~0.60/+0.050. `market_tree.svg` + `market_signal.svg` giờ KHÔNG dùng. Nhãn part-d trong notebook ôm cả §3d+§3e. Reconcile/anonymity/Word-template để CUỐI.

## 2. GIỌNG + CRAFT DẪN CẢM XÚC (chắt lọc từ RẤT nhiều vòng sửa session này)

**A. TRỰC DIỆN, KHÔNG VĂN VẺ (lỗi tái phạm #1).** Tiếng Việt mình chốt với user luôn rõ; tiếng Anh thì cứ trôi sang "văn". Chữa MỖI LẦN: câu ngắn; **mỗi đoạn MỘT nhịp**; **câu-neo đứng RIÊNG một dòng** (đừng chôn giữa câu). Bỏ mệnh đề phụ, bỏ hình ảnh điệu ("the ground the whole thing was standing on", "all that care buys you"). Giọng bạn-đồng-hành plain: *"But hang on… we did the right thing, we just did it once, and called it done."* **Test:** đọc riêng các dòng-neo từ trên xuống, phải tái hiện đúng chuỗi key tiếng Việt. Anh "văn" hơn Việt = SAI. TUYỆT ĐỐI KHÔNG em-dash (—).

**B. CÙNG-KHÁM-PHÁ, KHÔNG BIẾT TRƯỚC.** Mỗi nước đi thúc bởi cái thấy trước mắt; để DATA bung bất ngờ. Kinh nghiệm-mang-theo (sẹo chương trước) ≠ biết-trước-bẫy-chương-này.

**C. MỖI PHẦN: chốt KEY + PHẢN ỨNG KỲ VỌNG trước khi viết.** User lái bằng cảm xúc, nên trước mỗi phần phải nói rõ: *ý cốt lõi một câu* + *người đọc sẽ cảm/nghĩ gì*. Rồi mới viết phục vụ đúng câu neo đó. Số nhẹ tay — HÌNH gánh con số, prose gánh cảm xúc.

**D. ẨN DỤ ĐỜI THƯỜNG, CHẠY SUỐT, KHÔNG ĐỔI GIỮA CHỪNG.** Mỗi khái niệm khó = một ví dụ đời thường (Loan: bác sĩ lười; §3c: flashcard ôn-thi; §3d: ly cà phê/"ngưỡng bình thường cũ"; §3e: **học sinh/bài thi/bạn lười**). LỖI hay mắc: kể bằng ẩn dụ rồi nhảy về jargon (Frame/Station) → reader rớt mạch. Chữa: **bắc cầu** ("Frame = station một = ra đề"), giữ ẩn dụ chạy suốt.
- *Đặc biệt: "lazy guesser" KHÔNG phải máy 50:50.* Nó **luôn nói lớp đông** → điểm nó TRÔI theo độ lệch → đó mới là engine của "cái vạch trôi". Phải nói thẳng "not a coin".

**E. DẪN BẰNG CHUỖI CÂU HỎI ở chỗ vỡ sâu nhất.** Đáy cảm xúc (§3e Part 3) đi THẬT CHẬM, mỗi bước một câu hỏi nghi vấn ("So how? Where did it come from? Did we not check that one? Why not?") → phát hiện thành CỦA người đọc, không phải tác giả bảo.

**F. BẢN ĐỒ 3 NGHI CAN (device rất hiệu quả).** Sau khi ra số đẹp: *"nếu số này nói dối, nó chỉ trốn được ở mấy chỗ"* → liệt kê nghi can (khớp trạm pipeline) → soi từng cái → clear. Cú lật: cái tưởng đã clear (hoặc cái CHƯA BAO GIỜ lên danh sách) mới là thủ phạm. Hình: pipeline-với-chip (market_suspects), pipeline-mọc-nhánh-GUILTY (market_verdict/market_scale).

**G. LƯỢT-HAI SOI GƯƠNG LƯỢT-MỘT (cho bẫy lớn nhất).** "Giờ hiểu rồi, làm lại cho đúng" = **lặp Y HỆT** động tác lập-plan (§Xa) + soi-nghi-can (§Xb), có thể lặp câu chữ — sự đối xứng CHÍNH LÀ cách thể hiện "có kinh nghiệm thì xử lý sao". Cú đau: danh sách nghi can DÀI RA theo kinh nghiệm (3→4) mà bẫy sâu nhất VẪN không lên danh sách, vì nó không giống một bước — nó là "cái đề". → *"a choice we never even saw as a choice."*

**H. TÊN GIẢ ĐỊNH NẰM TRONG CÔNG THỨC, không phải cái núm bề mặt.** §3d không phải "percent vs points" (đọc thành "dùng %"); mà là **frozen mean/std = cược rằng thế giới đứng yên**; points chỉ là PHÉP THỬ phơi cược; percent CŨNG trôi (chỉ ít hơn), ta chưa bao giờ đo.

**I. LUẬT LEO THANG.** Khi bẫy mới TRÔNG GIỐNG bài học cũ: narrator phải ĐÃ ÁP đúng bài cũ mà VẪN ngã (vì bài cũ tự nó có giả định ngầm). Nói THẲNG ra: *"that check was right, and it was still not enough."* (§3e: đã kiểm cân bằng §2 đúng, nhưng đếm-một-lần-cho-cả-lịch-sử nên vẫn thiếu.)

**J. KHÔNG LỘ Ở TIÊU ĐỀ.** Tiêu đề đừng báo trước bẫy. Ở mục bẫy-lớn-nhất, tiêu đề phải RU NGỦ (hứa phần thưởng), vd. §3e "The process we can finally trust" (mỉa mai khi đọc lại). Đặt kỹ thuật mới phải có CONTEXT ("same honest test at 5 moments — no new trick"), không lôi chiêu lạ ra bất ngờ.

**K. Cấu trúc §3 làm mẫu:** a Reading (config + neo niềm tin + level-up ở bước MEASURE canh vết-thương-§2) → b By the book (số đẹp + 3 nghi can, tin) → c So was it the shuffle? (traceback → near-twin) → d And what about the scaling? (frozen mean/std) → e The process we can finally trust (3 movement: rebuild+tin → result+lộ "no single number"+lazy guesser → biggest assumption Frame) → f hạ cánh. Câu chốt chương (deepen §2): *"Trust the process, not the score. But what if the frame and the goal are unclear? What is the cleanest process worth then?"*

## 3. KỸ THUẬT
- **Python:** `py -3.14` (3.14, có numpy/matplotlib/nbformat/nbconvert). `.venv` HỎNG — đừng dùng.
- **Dựng notebook:** viết builder ở scratchpad (nbformat + `ExecutePreprocessor(kernel_name="python3")`, `resources={"metadata":{"path": NB_DIR}}`), construct hết cell rồi execute để lưu output THẬT. Chart inline: đầu cell `%matplotlib inline` + `%config InlineBackend.figure_format='svg'`; mỗi hình `fig.savefig(f"{FIG}/x.svg"); plt.show()`; `FIG = "figures" if os.path.isdir("figures") else "../figures"`. Warning zmq Proactor = vô hại.
- **Soi hình:** browser pane render file:// KHÔNG tin cậy (snapshot cũ). Cách chắc: render matplotlib ra PNG bằng script Agg rồi Read PNG. SVG vẽ tay thì không auto-render được ở đây — nhờ user liếc.
- **Palette:** accent `#3a6ea5`, muted `#b8c0cc`, đỏ `#c85a52`, grid `#eef1f4`; spine sạch. SVG vẽ tay: nền trắng `<rect fill="#ffffff">`, font Segoe UI.
- **Config B (mẫu, dùng SHAPE tương tự chương khác trừ khi cần khác):** train/val/test 60/20/20 (val để dành, §3 không dùng), standardize fit-train-only, MLP feats→16 ReLU→2 softmax, cross-entropy, plain GD lr **0.3**, 300 epoch, seeds 0–4. Market cần chronological split + rolling scaler + walk-forward. **Market DÙNG CHUNG 60/20/20 như §2** (đừng đổi 80/20; "không val" đọc thành gài-hụt).
- **USER SỬA TAY file:** user thường tự chỉnh READ_ME giữa chừng → **LUÔN Read lại vùng liên quan trước khi Edit** (đừng ghi đè sửa của user).

## 4. Số canonical §3 (từ notebook — dùng để reconcile §6/§7)
- Leak: shuffle **0.612**, chronological **0.587**, walk-forward **0.601**; direction control **0.540 / 0.539** (gap ~0); near-twin chung **4/5**.
- Drift: percent+frozen **0.587**, points+frozen **0.509**, points+rolling **0.528**; frozen-normal percent 0.85%→0.75% (×0.9), points 10.6→37 (×3.5); test |z| percent 0.59/9.25, points **2.73/43.7**.
- Eras (walk-forward): **0.519 / 0.602 / 0.651 / 0.654 / 0.582**; busy share trôi **0.374–0.541**; margin vs bar-của-era **+0.050** (biên −0.025 → +0.127) so với headline **+0.112**; swing ≈ 9× nhiễu mẫu.
- Learning: chưa-train ~0.494 → settled ~0.630; test busy 45%/calm 55%, đúng 56% busy / 66% calm.

## 5. NHIỆM VỤ NGAY: §3f
Viết lại §3f ("What we can actually stand behind") cho khớp giọng plain/exam của §3e, và **hạ cánh vào câu chốt** *"...what if the frame and the goal are unclear? What is the cleanest process worth then?"*. Ý §3f: (1) cái GÌ sống sót (kỹ năng thật nhưng nhỏ, +5 điểm, thắng 4/5 era) — đừng bi quan giả; (2) cái CHẾT = câu ta muốn viết ("the model is 61% accurate") — không tồn tại một con số; (3) thay bằng một câu nhỏ đứng-sau-được (data + feature + split + scaling + biên độ); (4) nâng §2: process sạch vô nghĩa nếu frame/goal chưa vững. Theo Mode C: đề 2–3 phương án, USER QUYẾT, mẩu nhỏ, KHÔNG commit git.

## 6. Nhắc cuối
Luôn hỏi user trước mỗi lựa chọn logic (đề 2–3 phương án). Soi NORTH_STAR + memory mỗi mẩu. User cực chú trọng: **giọng NHẸ + TRỰC DIỆN (không văn vẻ) · cùng-khám-phá thật · số THẬT từ notebook · ẩn dụ đời thường chạy suốt · dẫn cảm xúc bằng câu hỏi · câu-neo đứng riêng · mỗi mục kết bằng câu hỏi · không em-dash.**
