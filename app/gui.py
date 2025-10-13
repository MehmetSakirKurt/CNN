"""PySide6 GUI that ties together patient management, inference, and reporting."""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCalendarWidget,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .config import (
    CLASS_DISPLAY,
    DEFAULT_MODEL_VERSION,
    SEVERITY_ORDER,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from .database import DatabaseBackend, PatientInfo, create_database_manager
from .model_service import ModelService, PredictionResult


logger = logging.getLogger(__name__)


class LoginDialog(QDialog):
    """Simple dialog for username/password verification."""

    def __init__(
        self,
        expected_username: str,
        expected_password: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.expected_username = expected_username
        self.expected_password = expected_password
        self.setWindowTitle("Giriş Gerekiyor")
        self.setModal(True)

        form = QFormLayout(self)
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        form.addRow("Kullanıcı Adı", self.username_input)
        form.addRow("Parola", self.password_input)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c62828;")
        form.addRow(self.error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addWidget(buttons)

    def accept(self) -> None:  # type: ignore[override]
        username = self.username_input.text().strip()
        password = self.password_input.text()
        if username == self.expected_username and password == self.expected_password:
            super().accept()
        else:
            self.error_label.setText("Hatalı kullanıcı adı veya parola.")
            self.password_input.clear()
            self.password_input.setFocus()




def ensure_authenticated(parent: QWidget | None = None) -> bool:
    """Prompt for credentials when APP_USERNAME/PASSWORD are set."""

    expected_username = os.getenv("APP_USERNAME")
    expected_password = os.getenv("APP_PASSWORD")

    if not expected_username and not expected_password:
        return True
    if expected_username is None or expected_password is None:
        logger.warning("APP_USERNAME veya APP_PASSWORD eksik; doğrulama atlandı.")
        return True

    dialog = LoginDialog(expected_username, expected_password, parent)
    return dialog.exec() == QDialog.Accepted



class BrainSegmentationWindow(QMainWindow):
    """Main application window replicating the mock-up layout."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        database_url: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Brain Segmentation - Alzheimer Tespit Sistemi")
        self.resize(1280, 720)

        self.model_service = ModelService(model_path)
        self.logger = logging.getLogger("brain.gui")
        self.database: Optional[DatabaseBackend] = None
        self._pending_status: Optional[tuple[str, int]] = None

        self.selected_patient_id: Optional[str] = None
        self.selected_image: Optional[Path] = None
        self._last_prediction_payload: Optional[Dict[str, object]] = None

        try:
            self.database = create_database_manager(database_url)
            backend_name = (
                "Bellek" if getattr(self.database, "is_ephemeral", False) else "Neon/PostgreSQL"
            )
            self.logger.info("Veritabanı arka ucu: %s", backend_name)
            if getattr(self.database, "is_ephemeral", False):
                self._pending_status = (
                    "Veriler bu oturum boyunca bellekte tutuluyor (Neon yapılandırılmadı).",
                    8000,
                )
        except Exception as exc:
            # Allow the UI to open even if DB connection fails.
            print(f"[WARN] Database connection failed: {exc}")
            self.logger.exception("Database connection failed")
            self.database = None
            self._pending_status = (
                "Veritabanı bağlantısı kurulamadı; tahminler kalıcı olarak kaydedilmeyecek.",
                8000,
            )

        self._build_ui()
        if self._pending_status:
            message, timeout = self._pending_status
            self.status_bar.showMessage(message, timeout)
        self.refresh_patient_lists()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        tabs = QTabWidget()
        tabs.addTab(self._build_patient_tab(), "Hasta İncelemesi")
        tabs.addTab(self._build_screening_tab(), "Erken Teşhis")
        self.tabs = tabs
        self.setCentralWidget(tabs)

    def _build_patient_tab(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        layout.addWidget(self._build_patient_card(), stretch=2)
        layout.addWidget(self._build_imaging_panel(), stretch=3)
        layout.addWidget(self._build_result_panel(), stretch=2)
        return container

    def _build_screening_tab(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Patient list
        left_box = QGroupBox("Hastalar")
        left_layout = QVBoxLayout(left_box)
        self.patient_list = QListWidget()
        self.patient_list.itemSelectionChanged.connect(self._on_patient_selected_from_list)
        left_layout.addWidget(self.patient_list)

        # Center action
        center_box = QWidget()
        center_layout = QVBoxLayout(center_box)
        center_layout.addStretch(1)
        self.switch_to_patient_tab_btn = QPushButton("Tarama")
        self.switch_to_patient_tab_btn.clicked.connect(
            lambda: self.tabs.setCurrentIndex(0)
        )
        self.switch_to_patient_tab_btn.setFixedWidth(180)
        center_layout.addWidget(self.switch_to_patient_tab_btn, 0, Qt.AlignHCenter)
        center_layout.addStretch(1)

        # Right risk panel
        right_box = QGroupBox("Erken Teşhis")
        right_layout = QVBoxLayout(right_box)
        self.risk_list = QListWidget()
        self.risk_list.setAlternatingRowColors(True)
        right_layout.addWidget(self.risk_list)

        layout.addWidget(left_box, 3)
        layout.addWidget(center_box, 1)
        layout.addWidget(right_box, 2)
        return container

    def _build_patient_card(self) -> QWidget:
        card = QGroupBox("Kimlik Bilgileri")
        layout = QVBoxLayout(card)
        layout.setSpacing(8)

        avatar = QLabel("🧠")
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet("font-size: 48px;")
        layout.addWidget(avatar, 0, Qt.AlignHCenter)

        self.national_id_edit = self._create_line_edit("T.C. Kimlik No")
        self.first_name_edit = self._create_line_edit("Adı")
        self.last_name_edit = self._create_line_edit("Soyadı")
        self.birth_place_edit = self._create_line_edit("Doğum Yeri")

        layout.addWidget(self.national_id_edit)
        layout.addWidget(self.first_name_edit)
        layout.addWidget(self.last_name_edit)
        layout.addWidget(self.birth_place_edit)

        birth_row = QWidget()
        birth_layout = QHBoxLayout(birth_row)
        birth_layout.setContentsMargins(0, 0, 0, 0)
        birth_layout.setSpacing(6)
        birth_label = QLabel("Doğum Tarihi")
        self.birth_date_edit = QDateEdit()
        self.birth_date_edit.setCalendarPopup(True)
        self.birth_date_edit.setDisplayFormat("dd/MM/yyyy")
        self.birth_date_edit.dateChanged.connect(self._update_age)
        birth_layout.addWidget(birth_label)
        birth_layout.addWidget(self.birth_date_edit)
        layout.addWidget(birth_row)

        age_row = QWidget()
        age_layout = QHBoxLayout(age_row)
        age_layout.setContentsMargins(0, 0, 0, 0)
        age_layout.setSpacing(6)
        age_label = QLabel("Yaş")
        self.age_value_label = QLabel("-")
        self.age_value_label.setAlignment(Qt.AlignCenter)
        self.age_value_label.setStyleSheet("font-weight: bold;")
        age_layout.addWidget(age_label)
        age_layout.addWidget(self.age_value_label)
        layout.addWidget(age_row)

        gender_row = QWidget()
        gender_layout = QHBoxLayout(gender_row)
        gender_layout.setContentsMargins(0, 0, 0, 0)
        gender_layout.setSpacing(6)
        gender_label = QLabel("Cinsiyet")
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["-", "Kadın", "Erkek", "Diğer"])
        gender_layout.addWidget(gender_label)
        gender_layout.addWidget(self.gender_combo)
        layout.addWidget(gender_row)

        self.save_patient_btn = QPushButton("Hastayı Kaydet / Güncelle")
        self.save_patient_btn.clicked.connect(self._save_patient)
        layout.addWidget(self.save_patient_btn)

        self.load_images_btn = QPushButton("MR Görüntüsü Seç")
        self.load_images_btn.clicked.connect(self._load_images)
        layout.addWidget(self.load_images_btn)

        layout.addStretch(1)
        return card

    def _build_imaging_panel(self) -> QWidget:
        panel = QGroupBox("MR Görüntüsü")
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        self.image_label = QLabel("Görüntü Yüklenmedi")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(320, 320)
        self.image_label.setStyleSheet(
            "border: 1px solid #999; background-color: #f5f5f5; color: #666;"
        )
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.image_hint_label = QLabel("1 adet MR görüntüsü yükleyin")
        self.image_hint_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_hint_label)

        self.predict_btn = QPushButton("Tahmin Yap")
        self.predict_btn.clicked.connect(self._run_predictions)
        layout.addWidget(self.predict_btn, alignment=Qt.AlignCenter)
        return panel

    def _build_result_panel(self) -> QWidget:
        panel = QGroupBox("Segmentasyon Sonucu")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        self.calendar = QCalendarWidget()
        self.calendar.setGridVisible(True)
        layout.addWidget(self.calendar)

        self.prediction_title = QLabel("Sonuç")
        self.prediction_title.setAlignment(Qt.AlignCenter)
        self.prediction_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(self.prediction_title)

        self.prediction_body = QLabel("Henüz tahmin yapılmadı.")
        self.prediction_body.setWordWrap(True)
        self.prediction_body.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_body)

        self.prob_table = QTableWidget(0, 2)
        self.prob_table.setHorizontalHeaderLabels(["Sınıf", "Olasılık"])
        self.prob_table.horizontalHeader().setStretchLastSection(True)
        self.prob_table.verticalHeader().setVisible(False)
        self.prob_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.prob_table)

        self.export_csv_btn = QPushButton("CSV Olarak Dışa Aktar")
        self.export_csv_btn.clicked.connect(self._export_prediction_report)
        layout.addWidget(self.export_csv_btn)

        layout.addStretch(1)
        return panel

    def _create_line_edit(self, placeholder: str) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        return edit

    # ------------------------------------------------------------------ Patient actions
    def _update_age(self) -> None:
        qdate = self.birth_date_edit.date()
        if not qdate.isValid():
            self.age_value_label.setText("-")
            return
        today = date.today()
        bdate = qdate.toPython()
        years = today.year - bdate.year - (
            (today.month, today.day) < (bdate.month, bdate.day)
        )
        self.age_value_label.setText(str(years))

    def _collect_patient_info(self) -> Optional[PatientInfo]:
        national_id = self.national_id_edit.text().strip()
        first_name = self.first_name_edit.text().strip()
        last_name = self.last_name_edit.text().strip()

        if not (national_id and first_name and last_name):
            self.status_bar.showMessage("T.C. kimlik, ad ve soyad zorunludur.", 5000)
            return None

        birth_date = self.birth_date_edit.date().toPython() if self.birth_date_edit.date().isValid() else None
        gender = self.gender_combo.currentText()
        gender = None if gender == "-" else gender

        return PatientInfo(
            national_id=national_id,
            first_name=first_name,
            last_name=last_name,
            birth_place=self.birth_place_edit.text().strip() or None,
            birth_date=birth_date,
            gender=gender,
        )

    def _save_patient(self) -> None:
        if self.database is None:
            QMessageBox.warning(
                self,
                "Veritabanı Yok",
                "Neon/PostgreSQL bağlantısı yapılamadı. Lütfen DATABASE_URL ayarlayın.",
            )
            return

        info = self._collect_patient_info()
        if info is None:
            return

        try:
            patient_id = self.database.upsert_patient(info)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Kaydetme Hatası",
                f"Hasta kaydedilemedi:\n{exc}",
            )
            return

        self.selected_patient_id = patient_id
        self.status_bar.showMessage("Hasta bilgileri kaydedildi.", 5000)
        self.refresh_patient_lists()

    # ------------------------------------------------------------------ Imaging / predictions
    def _load_images(self) -> None:
        filters = " ".join(f"*{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "MR Görüntüsü Seç",
            "",
            f"Görüntü Dosyaları ({filters})",
        )
        if not file_path:
            return

        path = Path(file_path)
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.image_label.setText("Görüntü Yüklenemedi")
            self.image_label.setPixmap(QPixmap())
            self.status_bar.showMessage("Görüntü okunamadı, lütfen farklı bir dosya seçin.", 5000)
            return

        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setText("")
        self.image_hint_label.setText(path.name)
        self.selected_image = path

    def _run_predictions(self) -> None:
        if self.selected_image is None:
            QMessageBox.information(
                self,
                "Görüntü Yok",
                "Lütfen tahmin yapmadan önce bir MR görüntüsü seçin.",
            )
            return

        if self.database and self.selected_patient_id is None:
            # Attempt to save patient automatically if possible.
            info = self._collect_patient_info()
            if info:
                try:
                    self.selected_patient_id = self.database.upsert_patient(info)
                except Exception as exc:
                    self.status_bar.showMessage(f"Hasta kaydedilemedi: {exc}", 5000)

        try:
            result = self.model_service.predict(self.selected_image)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Tahmin Hatası",
                f"Görüntü işlenemedi:\n{exc}",
            )
            return

        if self.database and self.selected_patient_id:
            try:
                self.database.record_exam_result(
                    self.selected_patient_id,
                    image_path=str(self.selected_image),
                    prediction={
                        "label": result.label,
                        "confidence": result.confidence,
                        "probabilities": result.probabilities,
                    },
                    model_version=DEFAULT_MODEL_VERSION,
                )
            except Exception as exc:
                self.status_bar.showMessage(f"Sonuç kaydı yapılamadı: {exc}", 5000)

        self._last_prediction_payload = {
            "created_at": datetime.utcnow(),
            "patient_id": self.selected_patient_id,
            "image_path": str(self.selected_image) if self.selected_image else "",
            "predicted_label": result.label,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "model_version": getattr(self.model_service, "model_version", DEFAULT_MODEL_VERSION),
        }
        self.logger.info(
            "Tahmin tamamlandı: label=%s confidence=%.3f patient=%s image=%s",
            result.label,
            result.confidence,
            self.selected_patient_id or "-",
            self.selected_image or "-",
        )
        self._update_prediction_panel(result.label, result.confidence, result.probabilities)
        self.status_bar.showMessage("Tahmin tamamlandı.", 5000)
        self.refresh_patient_lists()

    def _export_prediction_report(self) -> None:
        records: List[Dict] = []
        if self.database and self.selected_patient_id:
            try:
                records = self.database.patient_history(self.selected_patient_id, limit=256)
            except Exception as exc:
                self.status_bar.showMessage(f"Hasta geçmişi alınamadı: {exc}", 5000)
                self.logger.exception("Patient history fetch failed")
                records = []
        if not records and self._last_prediction_payload:
            records = [self._last_prediction_payload]
        if not records:
            self.status_bar.showMessage("Dışa aktarılacak veri bulunamadı.", 5000)
            return

        suggested_name = "_".join(
            part for part in [self.first_name_edit.text().strip(), self.last_name_edit.text().strip(), "rapor"] if part
        ) or "rapor"
        suggested_path = str((Path.home() / f"{suggested_name}.csv").resolve())
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "CSV Raporunu Kaydet",
            suggested_path,
            "CSV Dosyaları (*.csv)",
        )
        if not file_path:
            return

        rows = self._build_export_rows(records)
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as handle:
                fieldnames = list(rows[0].keys())
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Dışa Aktarma Hatası",
                f"CSV kaydedilemedi:\n{exc}",
            )
            self.logger.exception("CSV export failed for %s", file_path)
            return

        self.status_bar.showMessage("CSV dışa aktarımı tamamlandı.", 6000)
        self.logger.info("CSV raporu dışa aktarıldı: %s (%d satır)", file_path, len(rows))

    def _build_export_rows(self, records: List[Dict]) -> List[Dict[str, str]]:
        base = {
            "patient_id": self.selected_patient_id or "",
            "national_id": self.national_id_edit.text().strip(),
            "first_name": self.first_name_edit.text().strip(),
            "last_name": self.last_name_edit.text().strip(),
        }
        class_labels = list(self.model_service.classes)
        rows: List[Dict[str, str]] = []
        for record in records:
            row = dict(base)
            created = record.get("created_at")
            if isinstance(created, datetime):
                timestamp = created.isoformat()
            else:
                timestamp = str(created or datetime.utcnow().isoformat())
            row["timestamp"] = timestamp
            row["model_version"] = str(record.get("model_version") or getattr(self.model_service, "model_version", DEFAULT_MODEL_VERSION))
            row["image_path"] = str(record.get("image_path") or "")
            row["predicted_label"] = str(record.get("predicted_label") or "")
            conf = record.get("confidence")
            row["confidence"] = f"{conf:.4f}" if isinstance(conf, (int, float)) else ""

            probabilities = record.get("probabilities") or {}
            if isinstance(probabilities, str):
                try:
                    probabilities = json.loads(probabilities)
                except json.JSONDecodeError:
                    probabilities = {}
            for cls in class_labels:
                value = probabilities.get(cls, 0.0)
                value = value if isinstance(value, (int, float)) else 0.0
                row[f"prob_{cls}"] = f"{value:.4f}"
            rows.append(row)
        return rows


    def _update_prediction_panel(
        self, label: str, confidence: float, probabilities: Dict[str, float]
    ) -> None:
        self.prediction_title.setText("Tahmin Sonucu")
        alzheimer_probability = 1.0 - probabilities.get("No Impairment", 0.0)
        label_tr = CLASS_DISPLAY.get(label, label)
        if label == "No Impairment":
            summary = (
                f"Model görüntüyü %{probabilities.get('No Impairment', 0.0) * 100:.1f} "
                "olasılıkla sağlıklı olarak etiketledi."
            )
        else:
            summary = (
                f"Model, bu MR görüntüsünün %{confidence * 100:.1f} olasılıkla {label_tr} "
                "olduğunu tahmin ediyor."
            )
        summary += f"\nAlzheimer riski (No Impairment dışı sınıfların toplamı): %{alzheimer_probability * 100:.1f}"
        self.prediction_body.setText(summary)

        self.prob_table.setRowCount(len(probabilities))
        for row, (cls, prob) in enumerate(sorted(probabilities.items(), key=lambda x: -x[1])):
            cls_display = CLASS_DISPLAY.get(cls, cls)
            cls_item = QTableWidgetItem(cls_display)
            prob_item = QTableWidgetItem(f"%{prob * 100:.1f}")
            if cls == label:
                cls_item.setForeground(QColor("#c62828"))
                prob_item.setForeground(QColor("#c62828"))
                cls_font = cls_item.font()
                cls_font.setBold(True)
                cls_item.setFont(cls_font)
                prob_font = prob_item.font()
                prob_font.setBold(True)
                prob_item.setFont(prob_font)
            self.prob_table.setItem(row, 0, cls_item)
            self.prob_table.setItem(row, 1, prob_item)

    # ------------------------------------------------------------------ Screening tab helpers
    def refresh_patient_lists(self) -> None:
        if self.database is None:
            return
        try:
            patients = self.database.list_patients()
        except Exception as exc:
            self.status_bar.showMessage(f"Hasta listesi alınamadı: {exc}", 5000)
            return

        self.patient_list.clear()
        for idx, patient in enumerate(patients, start=1):
            full_name = f"{patient['first_name']} {patient['last_name']}"
            item = QListWidgetItem(f"{idx}) {full_name}")
            item.setData(Qt.UserRole, patient)
            self.patient_list.addItem(item)

        # Build risk list
        risky = [
            p
            for p in patients
            if p.get("predicted_label") and SEVERITY_ORDER.get(p["predicted_label"], 0) > 0
        ]
        risky.sort(
            key=lambda p: (
                -SEVERITY_ORDER.get(p.get("predicted_label"), 0),
                -(p.get("confidence") or 0),
            )
        )
        self.risk_list.clear()
        for patient in risky[:5]:
            full_name = f"{patient['first_name']} {patient['last_name']}"
            label = patient.get("predicted_label", "Bilinmiyor")
            label_display = CLASS_DISPLAY.get(label, label)
            confidence = patient.get("confidence", 0) or 0
            item = QListWidgetItem(f"{full_name} - {label_display} (%{confidence * 100:.0f})")
            item.setData(Qt.UserRole, patient)
            item.setBackground(QColor("#ef9a9a"))
            self.risk_list.addItem(item)

    def _on_patient_selected_from_list(self) -> None:
        items = self.patient_list.selectedItems()
        if not items:
            return
        patient = items[0].data(Qt.UserRole)
        if not patient:
            return

        self.selected_patient_id = patient["patient_id"]
        self.national_id_edit.setText(patient["national_id"])
        self.first_name_edit.setText(patient["first_name"])
        self.last_name_edit.setText(patient["last_name"])
        self.birth_place_edit.setText(patient.get("birth_place") or "")
        if patient.get("birth_date"):
            qdate = QDate.fromString(str(patient["birth_date"]), "yyyy-MM-dd")
            if qdate.isValid():
                self.birth_date_edit.setDate(qdate)
        else:
            self.birth_date_edit.setDate(QDate.currentDate())
        gender = patient.get("gender") or "-"
        idx = self.gender_combo.findText(gender)
        self.gender_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.tabs.setCurrentIndex(0)
        self._update_age()


def launch_app(model_path: Path | str, database_url: str | None = None) -> None:
    """Entry point used by the CLI runner."""
    app = QApplication(sys.argv)
    if not ensure_authenticated():
        print("[INFO] Kullanıcı girişi iptal edildi.")
        return
    window = BrainSegmentationWindow(model_path=model_path, database_url=database_url)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_app(Path("alzheimer_cnn_torch.pt"))










