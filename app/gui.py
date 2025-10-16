from __future__ import annotations

import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QColor, QCursor, QPixmap
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
    QFrame,
    QHeaderView,
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


@dataclass(frozen=True)
class ComparisonResultRow:

    model_path: Path
    label: str
    confidence: float
    alzheimer_risk: float
    version: str
    arch: str


class LoginDialog(QDialog):

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

    def __init__(
        self,
        model_path: Path | str,
        *,
        database_url: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Alzheimer MR Model Stüdyosu")
        self.resize(1360, 840)
        self._install_base_styles()

        self.models_dir = Path("modeller")
        self.models_dir.mkdir(exist_ok=True)

        self.available_models: List[Path] = []
        self.selected_model_path: Optional[Path] = None
        self.model_service: Optional[ModelService] = None
        self.compare_image_path: Optional[Path] = None
        self.comparison_results: List[ComparisonResultRow] = []

        self.logger = logging.getLogger("brain.gui")
        self.database: Optional[DatabaseBackend] = None
        self._pending_status: Optional[tuple[str, int]] = None

        self.selected_patient_id: Optional[str] = None
        self.selected_image: Optional[Path] = None
        self._last_prediction_payload: Optional[Dict[str, object]] = None

        self.available_models = list(self.models_dir.glob("*.pt"))
        self.available_models.sort()

        try:
            self.database = create_database_manager(database_url)
            backend_name = (
                "Bellek"
                if getattr(self.database, "is_ephemeral", False)
                else "Neon/PostgreSQL"
            )
            self.logger.info("Veritabanı arka ucu: %s", backend_name)
            if getattr(self.database, "is_ephemeral", False):
                self._pending_status = (
                    "Veriler bu oturum boyunca bellekte tutuluyor (Neon yapılandırılmadı).",
                    8000,
                )
        except Exception as exc:
            print(f"[WARN] Database connection failed: {exc}")
            self.logger.exception("Database connection failed")
            self.database = None
            self._pending_status = (
                "Veritabanı bağlantısı kurulamadı; tahminler kalıcı olarak kaydedilmeyecek.",
                8000,
            )

        self._build_ui()

        self._update_model_list_ui()

        if self.available_models:
            self._load_model(self.available_models[0])

        if self._pending_status:
            message, timeout = self._pending_status
            self.status_bar.showMessage(message, timeout)
        self.refresh_patient_lists()

    def _build_ui(self) -> None:
        self.status_bar = QStatusBar(self)
        self.status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.status_bar)

        footer = QLabel("Dr. Öğr. Üyesi NURULLAH ÖZTÜRK")
        footer.setStyleSheet(
            "color: #475569; font-size: 12px; font-weight: 600; padding-left: 12px;"
        )
        self.status_bar.addPermanentWidget(footer)

        tabs = QTabWidget()
        tabs.setObjectName("mainTabs")
        tabs.setDocumentMode(True)
        tabs.setTabPosition(QTabWidget.North)
        tabs.addTab(self._build_patient_tab(), "Model Testi")
        tabs.addTab(self._build_screening_tab(), "Model Karşılaştırma")
        self.tabs = tabs
        self.setCentralWidget(tabs)

    def _install_base_styles(self) -> None:
        base_style = """
        QMainWindow {
            background-color: #f4f6fb;
        }
        QWidget {
            font-family: "Segoe UI", "Arial";
            color: #1f2933;
        }
        QStatusBar {
            background-color: #ffffff;
            border-top: 1px solid #d0d7de;
            padding: 6px 12px;
        }
        QTabWidget#mainTabs::pane {
            border: none;
        }
        QTabBar::tab {
            background-color: transparent;
            padding: 12px 18px;
            margin-right: 6px;
            border-radius: 12px;
        }
        QTabBar::tab:selected {
            background-color: #ffffff;
            color: #0f172a;
            font-weight: 600;
        }
        QTabBar::tab:!selected {
            color: #64748b;
        }
        QFrame#cardFrame {
            background-color: #ffffff;
            border: 1px solid #d0d7de;
            border-radius: 18px;
        }
        QLabel#sectionTitle {
            font-size: 22px;
            font-weight: 600;
        }
        QPushButton {
            border: none;
        }
        QPushButton[variant="primary"] {
            background-color: #2563eb;
            color: #ffffff;
            border-radius: 12px;
            padding: 10px 20px;
            font-weight: 600;
        }
        QPushButton[variant="primary"]:hover {
            background-color: #1d4ed8;
        }
        QPushButton[variant="primary"]:pressed {
            background-color: #1e40af;
        }
        QPushButton[variant="primary"]:disabled {
            background-color: #94a3b8;
            color: #e2e8f0;
        }
        QPushButton[variant="secondary"] {
            background-color: #ffffff;
            color: #1d4ed8;
            border: 1px solid #cbd5f5;
            border-radius: 12px;
            padding: 9px 18px;
            font-weight: 600;
        }
        QPushButton[variant="secondary"]:hover {
            background-color: #eef2ff;
        }
        QPushButton[variant="ghost"] {
            background-color: #f8fafc;
            color: #475569;
            border: 1px dashed #cbd5f5;
            border-radius: 12px;
            padding: 9px 16px;
            font-weight: 500;
        }
        QPushButton[variant="ghost"]:hover {
            background-color: #ffffff;
        }
        QListWidget#modelList,
        QListWidget#compareModelList {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 6px;
            background-color: #f8fafc;
        }
        QListWidget#modelList::item,
        QListWidget#compareModelList::item {
            padding: 10px 12px;
            margin: 3px 2px;
            border-radius: 8px;
        }
        QListWidget#modelList::item:selected,
        QListWidget#compareModelList::item:selected {
            background-color: #e0ecff;
            color: #1d4ed8;
        }
        QTableWidget#probabilityTable,
        QTableWidget#comparisonTable {
            background-color: transparent;
            border: none;
            gridline-color: #e2e8f0;
        }
        QHeaderView::section {
            background-color: #f1f5f9;
            color: #334155;
            padding: 8px;
            border: none;
            font-weight: 600;
        }
        """
        self.setStyleSheet(base_style)

    def _build_patient_tab(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(32, 28, 32, 28)
        outer.setSpacing(18)

        header_layout = QVBoxLayout()
        header_layout.setSpacing(6)

        title = QLabel("Model Testi")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignLeft)
        header_layout.addWidget(title)

        subtitle = QLabel("Modelleri içe aktarın; " "MR görüntüsünü için tahmin alın.")
        subtitle.setStyleSheet("color: #475569; font-size: 13px;")
        subtitle.setWordWrap(True)
        header_layout.addWidget(subtitle)

        outer.addLayout(header_layout)

        body = QHBoxLayout()
        body.setSpacing(18)
        body.addWidget(self._build_model_selection_panel(), stretch=2)
        body.addWidget(self._build_imaging_panel(), stretch=3)
        body.addWidget(self._build_result_panel(), stretch=2)
        outer.addLayout(body)

        outer.addStretch(1)
        return container

    def _build_screening_tab(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(32, 28, 32, 28)
        outer.setSpacing(18)

        header_layout = QVBoxLayout()
        header_layout.setSpacing(6)

        title = QLabel("Model Karşılaştırması")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignLeft)
        header_layout.addWidget(title)

        subtitle = QLabel(
            "Aynı MR görüntüsü üzerinden farklı modellerin tahminlerini yan yana inceleyin. "
            "En yüksek güvene sahip modeli ve Alzheimer risk skorlarını hızlıca görün."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #475569; font-size: 13px;")
        header_layout.addWidget(subtitle)

        outer.addLayout(header_layout)

        control_card = QFrame()
        control_card.setObjectName("cardFrame")
        card_layout = QVBoxLayout(control_card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(18)

        control_layout = QHBoxLayout()
        control_layout.setSpacing(24)

        image_column = QVBoxLayout()
        image_column.setSpacing(10)

        image_title = QLabel("Karşılaştırma Görseli")
        image_title.setStyleSheet("font-weight: 600; font-size: 13px;")
        image_column.addWidget(image_title)

        image_description = QLabel(
            "Seçeceğiniz MR görüntüsü tüm modeller için ortak olarak kullanılacaktır."
        )
        image_description.setWordWrap(True)
        image_description.setStyleSheet("color: #64748b; font-size: 12px;")
        image_column.addWidget(image_description)

        self.compare_image_preview = QLabel("Görsel seçilmedi")
        self.compare_image_preview.setAlignment(Qt.AlignCenter)
        self.compare_image_preview.setFixedSize(220, 220)
        self.compare_image_preview.setStyleSheet(
            "border: 2px dashed #cbd5f5; background-color: #f8fafc; color: #94a3b8; "
            "border-radius: 16px; font-size: 12px;"
        )
        image_column.addWidget(self.compare_image_preview, alignment=Qt.AlignCenter)

        self.compare_image_hint = QLabel(
            "Bir dosya seçin veya Model Testi sekmesindeki görüntüyü kullanın."
        )
        self.compare_image_hint.setWordWrap(True)
        self.compare_image_hint.setAlignment(Qt.AlignCenter)
        self.compare_image_hint.setStyleSheet("color: #64748b; font-size: 11px;")
        image_column.addWidget(self.compare_image_hint)

        image_buttons = QHBoxLayout()
        image_buttons.setSpacing(8)
        self.compare_image_btn = QPushButton("MR Görüntüsü Seç")
        self.compare_image_btn.setProperty("variant", "secondary")
        self.compare_image_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.compare_image_btn.clicked.connect(self._select_comparison_image)
        image_buttons.addWidget(self.compare_image_btn)

        self.use_active_image_btn = QPushButton("Aktif görüntüyü kullan")
        self.use_active_image_btn.setProperty("variant", "ghost")
        self.use_active_image_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.use_active_image_btn.clicked.connect(self._use_active_image_for_comparison)
        image_buttons.addWidget(self.use_active_image_btn)

        image_column.addLayout(image_buttons)
        control_layout.addLayout(image_column, stretch=2)

        model_column = QVBoxLayout()
        model_column.setSpacing(10)

        model_title = QLabel("Karşılaştırılacak Modeller")
        model_title.setStyleSheet("font-weight: 600; font-size: 13px;")
        model_column.addWidget(model_title)

        self.model_count_label = QLabel(f"{len(self.available_models)} model bulundu")
        self.model_count_label.setStyleSheet("color: #64748b; font-size: 11px;")
        model_column.addWidget(self.model_count_label)

        self.compare_model_list = QListWidget()
        self.compare_model_list.setObjectName("compareModelList")
        self.compare_model_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.compare_model_list.setSpacing(4)
        model_column.addWidget(self.compare_model_list, stretch=1)

        selection_buttons = QHBoxLayout()
        selection_buttons.setSpacing(8)

        self.select_all_models_btn = QPushButton("Tümünü Seç")
        self.select_all_models_btn.setProperty("variant", "secondary")
        self.select_all_models_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.select_all_models_btn.clicked.connect(self._select_all_comparison_models)
        selection_buttons.addWidget(self.select_all_models_btn)

        self.clear_selection_btn = QPushButton("Seçimi Temizle")
        self.clear_selection_btn.setProperty("variant", "ghost")
        self.clear_selection_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.clear_selection_btn.clicked.connect(self._clear_comparison_selection)
        selection_buttons.addWidget(self.clear_selection_btn)
        selection_buttons.addStretch(1)

        model_column.addLayout(selection_buttons)
        control_layout.addLayout(model_column, stretch=3)

        card_layout.addLayout(control_layout)

        self.run_comparison_btn = QPushButton("Karşılaştırmayı Başlat")
        self.run_comparison_btn.setProperty("variant", "primary")
        self.run_comparison_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.run_comparison_btn.clicked.connect(self._run_model_comparison)
        card_layout.addWidget(self.run_comparison_btn, alignment=Qt.AlignRight)

        outer.addWidget(control_card)

        self.comparison_summary_label = QLabel(
            "Henüz karşılaştırma yapılmadı. Bir MR görüntüsü seçip en az bir model işaretleyin."
        )
        self.comparison_summary_label.setWordWrap(True)
        self.comparison_summary_label.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; "
            "padding: 16px; font-size: 12px; color: #1f2937;"
        )
        outer.addWidget(self.comparison_summary_label)

        self.comparison_empty_label = QLabel(
            "Karşılaştırma sonuçları burada görünecek. Önce sol taraftan bir görüntü ve model(ler) seçin."
        )
        self.comparison_empty_label.setAlignment(Qt.AlignCenter)
        self.comparison_empty_label.setWordWrap(True)
        self.comparison_empty_label.setStyleSheet("color: #94a3b8; padding: 40px;")
        outer.addWidget(self.comparison_empty_label)

        self.comparison_table = QTableWidget(0, 5)
        self.comparison_table.setObjectName("comparisonTable")
        self.comparison_table.setHorizontalHeaderLabels(
            ["Model", "Tahmin", "Güven", "Alzheimer Riski", "Versiyon / Mimari"]
        )
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.comparison_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Stretch
        )
        self.comparison_table.verticalHeader().setVisible(False)
        self.comparison_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.comparison_table.setAlternatingRowColors(False)
        self.comparison_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.comparison_table.setStyleSheet("QTableWidget::item { padding: 8px; }")
        self.comparison_table.hide()
        outer.addWidget(self.comparison_table)

        outer.addStretch(1)
        return container

    def _build_model_selection_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("cardFrame")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Modeller")
        title.setAlignment(Qt.AlignLeft)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        description = QLabel(
            "Eğittiğiniz .pt modellerini `modeller/` klasörüne bırakın. "
            "Listeden bir model seçtiğinizde otomatik olarak yüklenir."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #64748b; font-size: 12px;")
        layout.addWidget(description)

        self.model_list = QListWidget()
        self.model_list.setObjectName("modelList")
        self.model_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.model_list.itemClicked.connect(self._on_model_selected)
        self.model_list.setSpacing(4)
        layout.addWidget(self.model_list, stretch=1)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self.refresh_models_btn = QPushButton("Modelleri Yenile")
        self.refresh_models_btn.setProperty("variant", "secondary")
        self.refresh_models_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.refresh_models_btn.clicked.connect(self._refresh_model_list)
        button_row.addWidget(self.refresh_models_btn)
        button_row.addStretch(1)

        layout.addLayout(button_row)

        self.model_info_label = QLabel("Henüz bir model seçilmedi.")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setTextFormat(Qt.RichText)
        self.model_info_label.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; "
            "border-radius: 12px; padding: 14px; font-size: 12px;"
        )
        layout.addWidget(self.model_info_label)

        return panel

    def _build_imaging_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("cardFrame")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        title = QLabel("MR Görüntüsü & Tahmin")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        header_row.addWidget(title)
        header_row.addStretch(1)

        self.active_model_chip = QLabel("Seçili model: -")
        self.active_model_chip.setStyleSheet(
            "background-color: #eef2ff; color: #1e40af; font-weight: 600; font-size: 11px; "
            "padding: 4px 10px; border-radius: 999px;"
        )
        header_row.addWidget(self.active_model_chip)

        layout.addLayout(header_row)

        helper = QLabel(
            "Önce modelinizi seçin, ardından MR görüntüsü yükleyerek tahmin alabilirsiniz. "
            "Görüntü, modele uygun şekilde otomatik ölçeklendirilir."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: #64748b; font-size: 12px;")
        layout.addWidget(helper)

        self.image_label = QLabel("Görüntü seçilmedi")
        self.image_label.setObjectName("imagePreview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(420, 420)
        self.image_label.setStyleSheet(
            "border: 2px dashed #cbd5f5; background-color: #f8fafc; color: #94a3b8; "
            "border-radius: 18px; font-size: 13px;"
        )
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.image_hint_label = QLabel("Lütfen bir MR görüntüsü seçin.")
        self.image_hint_label.setAlignment(Qt.AlignCenter)
        self.image_hint_label.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(self.image_hint_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        button_row.addStretch(1)

        self.load_images_btn = QPushButton("MR Görüntüsü Seç")
        self.load_images_btn.setProperty("variant", "secondary")
        self.load_images_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.load_images_btn.clicked.connect(self._load_images)
        button_row.addWidget(self.load_images_btn)

        self.predict_btn = QPushButton("Tahmin Yap")
        self.predict_btn.setProperty("variant", "primary")
        self.predict_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.predict_btn.clicked.connect(self._run_predictions)
        self.predict_btn.setEnabled(False)
        button_row.addWidget(self.predict_btn)

        layout.addLayout(button_row)
        return panel

    def _build_result_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("cardFrame")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Tahmin Sonuçları")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        self.prediction_title = QLabel("Henüz tahmin yapılmadı")
        self.prediction_title.setAlignment(Qt.AlignCenter)
        self.prediction_title.setStyleSheet(
            "background-color: #f1f5f9; border-radius: 12px; padding: 12px; "
            "font-weight: 600; color: #1f2937;"
        )
        layout.addWidget(self.prediction_title)

        self.prediction_body = QLabel(
            "Bir model ve MR görüntüsü seçtiğinizde sonuçlar burada listelenecek."
        )
        self.prediction_body.setWordWrap(True)
        self.prediction_body.setAlignment(Qt.AlignLeft)
        self.prediction_body.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; "
            "padding: 14px; font-size: 12px;"
        )
        layout.addWidget(self.prediction_body)

        table_label = QLabel("Sınıf olasılıkları")
        table_label.setStyleSheet("font-weight: 600; font-size: 12px; color: #475569;")
        layout.addWidget(table_label)

        self.prob_table = QTableWidget(0, 2)
        self.prob_table.setObjectName("probabilityTable")
        self.prob_table.setHorizontalHeaderLabels(["Sınıf", "Olasılık"])
        self.prob_table.horizontalHeader().setStretchLastSection(True)
        self.prob_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.prob_table.verticalHeader().setVisible(False)
        self.prob_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.prob_table.setAlternatingRowColors(False)
        self.prob_table.setStyleSheet("QTableWidget::item { padding: 8px; }")
        layout.addWidget(self.prob_table)

        layout.addStretch(1)
        return panel

    def _create_line_edit(self, placeholder: str) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        return edit

    def _scan_available_models(self) -> None:
        self.available_models = list(self.models_dir.glob("*.pt"))
        self.available_models.sort()
        self._update_model_list_ui()

    def _update_model_list_ui(self) -> None:
        self.model_list.clear()
        if hasattr(self, "compare_model_list"):
            self.compare_model_list.clear()

        if not self.available_models:
            item = QListWidgetItem("Henüz model bulunamadı")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.model_list.addItem(item)
            self.model_info_label.setText(
                "• `modeller/` klasörüne `.pt` uzantılı model dosyalarını ekleyin.<br>"
                "• Dosya eklendikten sonra <b>Modelleri Yenile</b> butonuna tıklayın."
            )
            if hasattr(self, "compare_model_list"):
                placeholder = QListWidgetItem("Model bulunamadı")
                placeholder.setFlags(placeholder.flags() & ~Qt.ItemIsEnabled)
                self.compare_model_list.addItem(placeholder)
            self.selected_model_path = None
        else:
            for model_path in self.available_models:
                item_text = model_path.stem
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, model_path)
                item.setToolTip(str(model_path))
                self.model_list.addItem(item)

                if model_path == self.selected_model_path:
                    item.setSelected(True)
                if hasattr(self, "compare_model_list"):
                    compare_item = QListWidgetItem(item_text)
                    compare_item.setData(Qt.UserRole, model_path)
                    compare_item.setToolTip(str(model_path))
                    self.compare_model_list.addItem(compare_item)
                    if model_path == self.selected_model_path:
                        compare_item.setSelected(True)

        if hasattr(self, "model_count_label"):
            count = len(self.available_models)
            suffix = "model bulundu" if count else "model yok"
            self.model_count_label.setText(f"{count} {suffix}")

        self._update_active_model_chip()
        if hasattr(self, "predict_btn"):
            self.predict_btn.setEnabled(self.model_service is not None)

    def _refresh_model_list(self) -> None:
        self._scan_available_models()

    def _load_model(self, model_path: Path) -> None:
        try:
            self.model_service = ModelService(model_path)
            self.selected_model_path = model_path

            info_text = (
                f"<b>{model_path.stem}</b><br>"
                f"Sürüm: {self.model_service.model_version}<br>"
                f"Mimari: {self.model_service.arch}<br>"
                f"Sınıf sayısı: {len(self.model_service.classes)}<br>"
                f"Çalışan cihaz: {self.model_service.device}"
            )

            self.model_info_label.setText(info_text)
            self.status_bar.showMessage(f"Model yüklendi: {model_path.name}", 3000)
            self.logger.info(f"Model loaded: {model_path}")
            self._update_active_model_chip()
            self.predict_btn.setEnabled(True)

            for index in range(self.model_list.count()):
                item = self.model_list.item(index)
                if item.data(Qt.UserRole) == model_path:
                    self.model_list.setCurrentItem(item)
                    break

            if hasattr(self, "compare_model_list"):
                for index in range(self.compare_model_list.count()):
                    item = self.compare_model_list.item(index)
                    if item.data(Qt.UserRole) == model_path:
                        item.setSelected(True)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Model Yükleme Hatası",
                f"Model yüklenemedi:\n{exc}",
            )
            self.model_info_label.setText(f"<b>Model yüklenemedi:</b><br>{exc}")
            self.logger.exception(f"Failed to load model: {model_path}")
            self.model_service = None
            self.selected_model_path = None
            self._update_active_model_chip()
            self.predict_btn.setEnabled(False)

    def _on_model_selected(self, item: QListWidgetItem) -> None:
        model_path = item.data(Qt.UserRole)
        if not model_path:
            return
        if model_path != self.selected_model_path:
            self._load_model(model_path)

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
            self.status_bar.showMessage(
                "Görüntü okunamadı, lütfen farklı bir dosya seçin.", 5000
            )
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
        if self.model_service is None:
            QMessageBox.warning(
                self,
                "Model Yok",
                "Lütfen önce bir model seçin.",
            )
            return

        if self.selected_image is None:
            QMessageBox.information(
                self,
                "Görüntü Yok",
                "Lütfen tahmin yapmadan önce bir MR görüntüsü seçin.",
            )
            return

        try:
            self.status_bar.showMessage("Tahmin yapılıyor...", 0)
            result = self.model_service.predict(self.selected_image)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Tahmin Hatası",
                f"Görüntü işlenemedi:\n{exc}",
            )
            self.status_bar.clearMessage()
            return

        self._last_prediction_payload = {
            "created_at": datetime.utcnow(),
            "image_path": str(self.selected_image) if self.selected_image else "",
            "predicted_label": result.label,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "model_version": getattr(
                self.model_service, "model_version", DEFAULT_MODEL_VERSION
            ),
            "model_path": (
                str(self.selected_model_path) if self.selected_model_path else ""
            ),
        }

        self.logger.info(
            "Tahmin tamamlandı: label=%s confidence=%.3f model=%s image=%s",
            result.label,
            result.confidence,
            self.selected_model_path.name if self.selected_model_path else "-",
            self.selected_image or "-",
        )

        self._update_prediction_panel(
            result.label, result.confidence, result.probabilities
        )
        display_label = CLASS_DISPLAY.get(result.label, result.label)
        self.status_bar.showMessage(
            f"Tahmin tamamlandı · {display_label} · %{result.confidence * 100:.1f} güven",
            5000,
        )

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
        for row, (cls, prob) in enumerate(
            sorted(probabilities.items(), key=lambda x: -x[1])
        ):
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

    def _update_active_model_chip(self) -> None:
        if hasattr(self, "active_model_chip"):
            if self.selected_model_path:
                self.active_model_chip.setText(
                    f"Seçili model: {self.selected_model_path.stem}"
                )
            else:
                self.active_model_chip.setText("Seçili model: -")

    def _update_comparison_image_preview(self, path: Optional[Path]) -> bool:
        if not hasattr(self, "compare_image_preview"):
            return False
        if path is None:
            self.compare_image_preview.setPixmap(QPixmap())
            self.compare_image_preview.setText("Görsel seçilmedi")
            if hasattr(self, "compare_image_hint"):
                self.compare_image_hint.setText(
                    "Bir dosya seçin veya Model Testi sekmesindeki görüntüyü kullanın."
                )
            return False

        path = Path(path)
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.compare_image_preview.setPixmap(QPixmap())
            self.compare_image_preview.setText("Görsel yüklenemedi")
            if hasattr(self, "compare_image_hint"):
                self.compare_image_hint.setText(
                    "Dosya açılamadı. Lütfen farklı bir görüntü seçin."
                )
            return False

        scaled = pixmap.scaled(
            self.compare_image_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.compare_image_preview.setPixmap(scaled)
        self.compare_image_preview.setText("")
        if hasattr(self, "compare_image_hint"):
            self.compare_image_hint.setText(
                f"Seçilen dosya: {path.name} ({pixmap.width()}×{pixmap.height()} piksel)"
            )
        return True

    def _select_comparison_image(self) -> None:
        filters = " ".join(f"*{ext}" for ext in SUPPORTED_IMAGE_EXTENSIONS)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Karşılaştırma için MR Görüntüsü Seç",
            "",
            f"Görüntü Dosyaları ({filters})",
        )
        if not file_path:
            return

        path = Path(file_path)
        if self._update_comparison_image_preview(path):
            self.compare_image_path = path
            self.status_bar.showMessage("Karşılaştırma için görüntü seçildi.", 4000)
        else:
            self.compare_image_path = None

    def _use_active_image_for_comparison(self) -> None:
        if self.selected_image is None:
            QMessageBox.information(
                self,
                "Görüntü bulunamadı",
                "Model Testi sekmesinde kullanılabilir bir MR görüntüsü bulunamadı.",
            )
            return

        path = Path(self.selected_image)
        if not path.exists():
            QMessageBox.warning(
                self,
                "Dosya bulunamadı",
                "Seçilen MR görüntüsü artık mevcut değil. Lütfen yeniden seçin.",
            )
            return

        if self._update_comparison_image_preview(path):
            self.compare_image_path = path
            self.status_bar.showMessage(
                "Karşılaştırma için aktif MR görüntüsü kullanılıyor.", 4000
            )

    def _select_all_comparison_models(self) -> None:
        if not hasattr(self, "compare_model_list"):
            return
        for index in range(self.compare_model_list.count()):
            item = self.compare_model_list.item(index)
            if item.flags() & Qt.ItemIsSelectable:
                item.setSelected(True)

    def _clear_comparison_selection(self) -> None:
        if not hasattr(self, "compare_model_list"):
            return
        for index in range(self.compare_model_list.count()):
            item = self.compare_model_list.item(index)
            item.setSelected(False)

    def _run_model_comparison(self) -> None:
        if not hasattr(self, "compare_model_list"):
            return

        if self.compare_image_path is None:
            QMessageBox.information(
                self,
                "Görüntü seçilmedi",
                "Karşılaştırmaya başlamadan önce bir MR görüntüsü seçin.",
            )
            return

        selected_items = [
            self.compare_model_list.item(i)
            for i in range(self.compare_model_list.count())
            if self.compare_model_list.item(i).isSelected()
            and self.compare_model_list.item(i).flags() & Qt.ItemIsSelectable
        ]

        if not selected_items:
            QMessageBox.information(
                self,
                "Model seçilmedi",
                "Karşılaştırmak için en az bir model seçin.",
            )
            return

        image_path = Path(self.compare_image_path)
        results: List[ComparisonResultRow] = []
        errors: List[str] = []
        selected_resolved: Optional[Path] = None
        if self.selected_model_path is not None:
            try:
                selected_resolved = self.selected_model_path.resolve()
            except Exception:
                selected_resolved = self.selected_model_path

        self.status_bar.showMessage("Modeller karşılaştırılıyor...", 0)
        self.run_comparison_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            for item in selected_items:
                model_path_data = item.data(Qt.UserRole)
                if not model_path_data:
                    continue
                model_path = Path(model_path_data)
                try:
                    try:
                        model_resolved = model_path.resolve()
                    except Exception:
                        model_resolved = model_path

                    reuse_existing = (
                        selected_resolved is not None
                        and self.model_service is not None
                        and model_resolved == selected_resolved
                    )
                    service = (
                        self.model_service
                        if reuse_existing
                        else ModelService(model_path)
                    )

                    prediction = service.predict(image_path)
                    version = getattr(service, "model_version", "-")
                    arch = getattr(service, "arch", "-")
                    risk = 1.0 - prediction.probabilities.get("No Impairment", 0.0)

                    results.append(
                        ComparisonResultRow(
                            model_path=model_path,
                            label=prediction.label,
                            confidence=prediction.confidence,
                            alzheimer_risk=risk,
                            version=str(version),
                            arch=str(arch),
                        )
                    )
                except Exception as exc:
                    error_message = f"{model_path.name}: {exc}"
                    errors.append(error_message)
                    self.logger.exception(
                        "Model karşılaştırması sırasında hata: %s", error_message
                    )
        finally:
            self.run_comparison_btn.setEnabled(True)
            QApplication.restoreOverrideCursor()

        results.sort(key=lambda row: row.confidence, reverse=True)
        self.comparison_results = results
        self._update_comparison_table(results)

        if errors:
            QMessageBox.warning(
                self,
                "Karşılaştırma Tamamlandı",
                "Bazı modeller çalıştırılamadı:\n" + "\n".join(errors),
            )

    def _update_comparison_table(self, rows: List[ComparisonResultRow]) -> None:
        if not hasattr(self, "comparison_table"):
            return

        if not rows:
            self.comparison_table.hide()
            if hasattr(self, "comparison_empty_label"):
                self.comparison_empty_label.show()
            if hasattr(self, "comparison_summary_label"):
                self.comparison_summary_label.setText(
                    "Henüz karşılaştırma yapılmadı. Bir MR görüntüsü seçip en az bir model işaretleyin."
                )
            self.status_bar.showMessage("Karşılaştırma sonuç üretmedi.", 4000)
            return

        self.comparison_table.clearContents()
        self.comparison_table.setRowCount(len(rows))
        best_row = rows[0]
        highlight_fg = QColor("#1d4ed8")
        highlight_bg = QColor("#e0f2fe")

        for row_index, row in enumerate(rows):
            model_item = QTableWidgetItem(row.model_path.stem)
            model_item.setToolTip(str(row.model_path))

            label_display = CLASS_DISPLAY.get(row.label, row.label)
            label_item = QTableWidgetItem(label_display)

            confidence_item = QTableWidgetItem(f"%{row.confidence * 100:.1f}")
            confidence_item.setTextAlignment(Qt.AlignCenter)

            risk_item = QTableWidgetItem(f"%{row.alzheimer_risk * 100:.1f}")
            risk_item.setTextAlignment(Qt.AlignCenter)

            meta_item = QTableWidgetItem(f"{row.version} • {row.arch}")
            meta_item.setTextAlignment(Qt.AlignCenter)

            if row is best_row:
                for item in (
                    model_item,
                    label_item,
                    confidence_item,
                    risk_item,
                    meta_item,
                ):
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setForeground(highlight_fg)
                    item.setBackground(highlight_bg)

            self.comparison_table.setItem(row_index, 0, model_item)
            self.comparison_table.setItem(row_index, 1, label_item)
            self.comparison_table.setItem(row_index, 2, confidence_item)
            self.comparison_table.setItem(row_index, 3, risk_item)
            self.comparison_table.setItem(row_index, 4, meta_item)

        self.comparison_table.resizeRowsToContents()
        self.comparison_table.show()
        if hasattr(self, "comparison_empty_label"):
            self.comparison_empty_label.hide()

        summary_text = (
            f"En yüksek güven skoru: <b>{best_row.model_path.stem}</b> · "
            f"{CLASS_DISPLAY.get(best_row.label, best_row.label)} · "
            f"%{best_row.confidence * 100:.1f} güven.<br>"
            f"Alzheimer riski (No Impairment dışı sınıflar): %{best_row.alzheimer_risk * 100:.1f}"
        )
        if hasattr(self, "comparison_summary_label"):
            self.comparison_summary_label.setText(summary_text)

        self.status_bar.showMessage(
            f"{len(rows)} model karşılaştırıldı. En yüksek güven {best_row.model_path.stem} modelinde.",
            6000,
        )

    def refresh_patient_lists(self) -> None:
        pass


def launch_app(model_path: Path | str, database_url: str | None = None) -> None:
    app = QApplication(sys.argv)
    if not ensure_authenticated():
        print("[INFO] Kullanıcı girişi iptal edildi.")
        return
    window = BrainSegmentationWindow(model_path=model_path, database_url=database_url)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_app(Path("alzheimer_cnn_torch.pt"))

