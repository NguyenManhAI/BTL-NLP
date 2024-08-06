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
- có ví dụ cụ thể trong main.py