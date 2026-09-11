# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):

  {
  "class_id": 468,
  "class_name": "cab",
  "rank": 1,
  "score": 0.510915,
  "taxonomy_name": "ImageNet-1K"
  }

- Record này mô tả toàn ảnh như thế nào?

  Record này mô tả kết quả phân loại của toàn bộ ảnh, toàn bộ ảnh được mô hình phân loại là ảnh về 1 chiếc taxi (cab) với độ tin cậy khoảng 51,09%

- Ai định nghĩa class list mà checkpoint có thể dự đoán?

  class list được định nghĩa bởi dataset dùng để huấn luyện. Checkpoint được huấn luyện trên ImageNet-1K

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?

  Cần giữ cả ID, tên lớp và tên taxonomy là điều rất quan trọng bởi vì nếu có cùng một class_id nhưng ở hai taxonomy khác nhau thì có thể mang ý nghĩa khách nhau, tên lớp giúp con người hiểu kết quả

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?

  Ảnh có nhiều chủ thể, guideline cần quy định rõ các nhãn dán ví dụ như chọn đối tượng chiếm diện tích nhất hoặc là chọn đối tượng chính của ảnh hoặc đối tượng chính mà người chụp muốn nhấn mạnh...
  Nếu không quy định rõ ràng thì hai annotator có thể gán hai nhãn khác nhau cùng một ảnh làm cho việc đánh giá model thiếu sự nhất quán

- Vì sao model score không phải ground truth?
  Groud truth là nhãn đúng mà do con người gán nhãn theo guideline cồn model score chỉ là mức độ tin tưởng của mô hình đối với dự đoán.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  {
  "class_name": "person",
  "score": 0.912624,
  "bbox_xyxy": [
  385.33,
  69.24,
  498.92,
  348.92
  ],
  "bbox_width": 113.58,
  "bbox_height": 279.68
  }
- Diễn giải vị trí box bằng lời:

  Bounding box giúp chuyển thông tin từ dạng số sang vị trí trực quan, giúp reviewer biết object nằm ở đâu và có bao phủ đúng vật thể hay không.

- So sánh số prediction ở hai threshold:

  Threshold thấp: giữ lại nhiều prediction hơn, tăng khả năng phát hiện đầy đủ object nhưng có thể xuất hiện nhiều dự đoán sai, làm tăng khối lượng reviewer kiểm tra.
  Threshold cao: giảm số lượng prediction cần xem và tăng độ chính xác, nhưng có nguy cơ bỏ sót các object nhỏ hoặc khó nhận dạng

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?

  Khi giảm threshold, số lượng prediction tăng lên giúp tăng độ bao phủ (coverage), giảm khả năng bỏ sót các object nhỏ hoặc khó nhận dạng. Tuy nhiên, số lượng bounding box cần kiểm tra cũng tăng, làm tăng khối lượng công việc của reviewer.
  Ngược lại, khi tăng threshold, số prediction giảm giúp reviewer xử lý nhanh hơn và ít phải kiểm tra hơn, nhưng có thể làm giảm độ bao phủ do bỏ qua một số object có confidence thấp.

- Đề xuất một quy tắc box chặt:

  Box phải bao phủ đầy đủ phần nhìn thấy của object.
  Hạn chế chứa vùng nền không liên quan.
  Các box trùng lặp hoặc sai vị trí cần được loại bỏ.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?

  Với object bị che khuất hoặc cắt mép, cần có guideline để quyết định cách gán bounding box và trường hợp cần escalation.
  Nếu object vẫn nhận dạng được và phần nhìn thấy đủ thông tin, bounding box được tạo theo phần vật thể xuất hiện trong ảnh.
  Nếu object bị che khuất quá nhiều, không xác định rõ class hoặc ranh giới object thì cần đưa sang bước review/escalation.
  Với object bị cắt bởi mép ảnh, cần thống nhất chỉ đánh dấu phần vật thể nhìn thấy trong ảnh hoặc áp dụng quy tắc riêng tùy mục đích dataset.
  Các trường hợp không rõ ràng cần có reviewer cấp cao quyết định để đảm bảo tính nhất quán của dữ liệu.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  {
  "instance_id": "kitchen-001",
  "class_name": "person",
  "score": 0.899318,
  "bbox_xyxy": [385.45, 66.44, 498.02, 348.58],
  "polygon_point_count": 348,
  "polygon_xy": [...]
  }
- Polygon bổ sung chi tiết gì so với box?

  Bounding Box chỉ mô tả vùng hình chữ nhật bao quanh đối tượng thông qua tọa độ (x_min, y_min, x_max, y_max), giúp xác định vị trí và kích thước tổng quát của vật thể.
  Polygon bổ sung các thông tin mà Bounding Box không thể thể hiện:
  Mô tả chính xác hình dạng thực của đối tượng.
  Xác định pixel nào thuộc đối tượng và pixel nào thuộc nền.
  Tính được diện tích thực của đối tượng.
  Phân tách tốt các đối tượng chồng lấn hoặc nằm sát nhau.
  Hỗ trợ các tác vụ Instance Segmentation, đo lường và phân tích hình học.

- `instance_id` dùng để làm gì và không phải loại ID nào?

  instance_id là mã định danh duy nhất cho một đối tượng cụ thể trong ảnh
  Loại ID không phải:
  Không phải class_id (mã phân loại danh mục chung như 0 cho person, 45 cho bowl).
  Không phải coco_image_id (mã định danh toàn cục của cả bức ảnh).
  Không phải Global Tracking ID / Re-ID (không dùng để định danh duy nhất đối tượng đó xuyên suốt qua nhiều khung hình video hay các bức ảnh khác nhau).

- Đề xuất một quy tắc biên mask:

Quy tắc lấy ngưỡng pixel (Pixel Inclusion Rule): Một pixel nằm trên đường ranh giới sẽ được tính vào mask nếu ít nhất 50% diện tích pixel đó thuộc về thực thể.
Quy tắc làm mịn và đơn giản hóa (Simplification): Loại bỏ các đỉnh polygon dư thừa nằm trên cùng một đường thẳng hoặc khoảng cách giữa 2 điểm liên tiếp nhỏ hơn 1 pixel để giảm độ nhiễu giăng cưa (zig-zag).

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  Vùng mờ: Guideline cần định nghĩa rõ ngưỡng tương phản để cắt viền mask, loại bỏ phần bóng mờ quá nhạt không thuộc thân thể vật thể.
  Vùng tiếp xúc : Guideline cần quy định nguyên tắc không cho phép 2 mask của 2 instance tách biệt bị dính/chồng lấn lên nhau
  Vùng bị che khuất : Guideline cần xác định dự án chạy theo tiêu chí Visible-only (chỉ vẽ phần nhìn thấy) hay Amodal Segmentation (vẽ suy đoán cả phần bị che).
  Nội dung cần Escalation :
  Vật thể bị che khuất quá 80% diện tích và không rõ ranh giới.
  Vật thể bị nhòe mờ chuyển động nặng khiến ranh giới biến dạng hoàn toàn.
  Các trường hợp hai vật thể cùng màu sắc/chất liệu đè lên nhau mà mắt thường không thể phân định ranh giới tách biệt.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --------------------- | ----------------------------- | --------------------------------- | ----------------- | ---------------- |
| Phân loại ảnh         |                               |                                   |                   |                  |
| Phát hiện vật thể     |                               |                                   |                   |                  |
| Instance segmentation |                               |                                   |                   |                  |

1. Tác vụ Phân loại ảnh (Image Classification)
   Định dạng Ground Truth: Nhãn danh mục (Class ID/Class Label dưới dạng Text hoặc Integer) hoặc Vector xác suất (One-hot vector) đối với bài toán multi-class/multi-label.
   Lỗi và điểm mơ hồ thường gặp: Bối cảnh chứa đồng thời nhiều vật thể thuộc các lớp khác nhau; vật thể chính bị che khuất nghiêm trọng hoặc nhầm lẫn giữa các nhãn con có độ tương đồng cao.
   Nhiệm vụ của Annotator: Quan sát toàn bộ bức ảnh và gán một hoặc nhiều nhãn phân loại phù hợp nhất dựa trên tài liệu hướng dẫn (guideline).
   Nhiệm vụ của Reviewer: Đối soát nhãn đã gán với nội dung tổng thể của ảnh, kiểm tra tính tuân thủ quy tắc danh mục (taxonomy) và phát hiện các trường hợp mơ hồ để báo cáo lên cấp quản lý.
2. Tác vụ Phát hiện vật thể (Object Detection)
   Định dạng Ground Truth: Tọa độ khung Bounding Box [xmin, ymin, xmax, ymax] (hoặc dạng center/width/height) đi kèm với Class ID / Class Name.
   Lỗi và điểm mơ hồ thường gặp: Khung bao quanh quá rộng (chứa nhiều bối cảnh nền) hoặc quá hẹp (cắt xẻm vật thể); bỏ sót đối tượng (False Negative); mơ hồ ranh giới khung khi vật thể bị che khuất một phần.
   Nhiệm vụ của Annotator: Kéo khung hình chữ nhật (Bounding box) ôm sát từng vật thể trong ảnh và gán nhãn phân loại tương ứng.
   Nhiệm vụ của Reviewer: Kiểm tra độ ôm sát (tightness) của từng khung, rà soát đối tượng bị bỏ sót, phát hiện khung gán sai nhãn hoặc bị trùng lặp.
3. Tác vụ Instance Segmentation
   \Định dạng Ground Truth: Chuỗi tọa độ điểm Polygon [[x1, y1], [x2, y2], ...], mã hóa RLE (Run-Length Encoding) hoặc Binary Mask, đi kèm Class ID và Instance ID riêng biệt.
   Lỗi và điểm mơ hồ thường gặp: Đường biên bị răng cưa, thiếu điểm ảnh; ranh giới giữa hai vật thể chạm nhau bị dính/chồng lấn; khó xác định vùng biên do bóng râm, nhòe chuyển động (motion blur) hoặc hiệu ứng trong suốt.
   Nhiệm vụ của Annotator: Vẽ chuỗi điểm Polygon chi tiết bám sát đường viền thực tế của từng thể hiện vật thể, chọn nhãn lớp và gán ID định danh riêng biệt cho từng đối tượng.
   Nhiệm vụ của Reviewer: Đánh giá độ chính xác cấp pixel của đường biên (Pixel-level IoU), kiểm tra lỗi dính/chồng lấn ranh giới giữa các instance và nghiệm thu việc xử lý vùng bị che khuất hoặc làm mờ.

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:

  Tuyệt đối bảo mật: Không sao chép, chụp ảnh màn hình, tải về máy cá nhân hay chia sẻ dữ liệu ra ngoài.
  Đúng phạm vi: Chỉ truy cập và xử lý đúng dữ liệu được giao nhiệm vụ.
  Khóa thiết bị: Luôn khóa màn hình khi rời vị trí, không dùng chung tài khoản.
  Báo cáo sự cố: Dừng thao tác và báo ngay cho Team Lead / PM khi phát hiện dữ liệu bất thường hoặc sai phạm vi.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

  Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho Team Lead hoặc Project Manager

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
