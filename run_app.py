
import sys
from pathlib import Path

# app modülünü içe aktarmak için yolu ekle
sys.path.insert(0, str(Path(__file__).parent))

from app.gui import launch_app


if __name__ == "__main__":
    # Varsayılan bir model yolu (kullanılmayacak ama parametre gerekli)
    default_model = Path("modeller/alzheimer_cnn_run1.pt")
    
    print("=" * 60)
    print("🧠 Alzheimer MR Görüntü Analiz Sistemi")
    print("=" * 60)
    print()
    print("📁 Modeller klasörü kontrol ediliyor...")
    
    models_dir = Path("modeller")
    models_dir.mkdir(exist_ok=True)
    
    model_files = list(models_dir.glob("*.pt"))
    
    if model_files:
        print(f"✅ {len(model_files)} model bulundu:")
        for model in model_files:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"   - {model.name} ({size_mb:.1f} MB)")
    else:
        print("⚠️  Henüz model yok!")
        print()
        print("📝 Lütfen 'modeller' klasörüne .pt uzantılı model dosyaları ekleyin.")
        print()
        print("Model eğitmek için:")
        print("   python train_model.py --model-path modeller/my_model.pt")
        print()
        print("Uygulama yine de açılacak, modelleri daha sonra ekleyebilirsiniz.")
    
    print()
    print("🚀 Uygulama başlatılıyor...")
    print()
    
    # GUI'yi başlat
    launch_app(model_path=default_model)

