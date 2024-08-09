# Dịch máy song ngữ Anh - Việt
## Hướng dẫn:
### Tổng quan:
- Toàn bộ chương trình cài đặt đều nằm trong tệp BTL
- Trong BTL có 4 tệp Translation - mỗi tệp là nơi đào tạo và kiểm tra mô hình
- Một tệp data lưu trữ dữ liệu song ngữ 
- Trong mỗi tệp translation gồm có:
    + một tệp train.py
    + một tệp predict.py
    + các tệp còn lại là kết quả của quá trình huấn luyện và kiểm tra
- Module dịch máy chính thức, chứa tất cả các mô hình dịch khác nhau: main.py
### Sử dụng dịch:
- class TranslationBiEnVi, khai báo và gọi phương thức translation để dịch
- đã thêm transformer vào trong phương thức translation, đầu vào, đầu ra giữ nguyên:
    - kind: thêm 1 biến transformer
    - model cũng như vocab cần để trong folder model (cùng vị trí với main.py)
    - Link model, vocab: [Model+Vocab](https://drive.google.com/drive/folders/1khpuzDEl6e0j7I6-nsN1YMR-Me7B2t6W?usp=sharing)
- có ví dụ cụ thể trong main.py