# 🛒 ARCart

**Walk in. Grab stuff. Walk out.** 

An AR shopping app for smart glasses that automatically tracks what you pick up, adds it to a virtual cart, and handles payment when you leave the store.

## 🎯 Vision

Imagine walking into any store with AR glasses:
1. **Pick up an item** → Camera recognizes it, adds to your virtual cart
2. **Put it back** → Automatically removed from cart
3. **Walk out** → Auto-checkout, receipt sent to your phone

No scanning. No checkout lines. No friction.

## 🔧 Technical Approach

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                      AR GLASSES                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Camera    │  │  AR Display │  │  Hand Tracking      │  │
│  │  (product   │  │  (cart UI,  │  │  (pick up/put down  │  │
│  │   recog)    │  │   prices)   │  │   detection)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    MOBILE APP / BACKEND                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Product   │  │   Virtual   │  │   Payment           │  │
│  │   Database  │  │   Cart      │  │   Processing        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack (Planned)

| Component | Technology | Notes |
|-----------|------------|-------|
| AR Platform | WebXR / Meta SDK / Apple ARKit | Cross-platform support |
| Product Recognition | TensorFlow Lite / YOLO | On-device ML for speed |
| Barcode Fallback | ZXing / ML Kit | When visual recognition fails |
| Backend | Node.js + PostgreSQL | Product database, cart sync |
| Payments | Stripe / Apple Pay | Seamless checkout |
| Mobile Companion | React Native | Cart review, receipts, history |

### Recognition Strategies

1. **Visual Product Recognition** (primary)
   - ML model trained on product images
   - Works for unique-looking items

2. **Barcode/QR Scanning** (fallback)
   - Standard UPC/EAN codes
   - Store-specific QR codes

3. **Shelf Location** (enhancement)
   - Know what products are where
   - Helps narrow down recognition

4. **Price Tag OCR** (verification)
   - Read price tags to confirm product
   - Catch sale prices

## 🎮 User Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Enter Store │ ──▶ │  Auto-detect │ ──▶ │  Start       │
│              │     │  store       │     │  Session     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Item Added  │ ◀── │  Recognize   │ ◀── │  Pick Up     │
│  to Cart     │     │  Product     │     │  Item        │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                         │
       │         ┌──────────────┐                │
       │         │  Put Back?   │ ◀──────────────┘
       │         │  Remove from │
       │         │  Cart        │
       │         └──────────────┘
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Exit Store  │ ──▶ │  Auto        │ ──▶ │  Receipt     │
│  (geofence)  │     │  Checkout    │     │  Sent        │
└──────────────┘     └──────────────┘     └──────────────┘
```

## 📱 AR Display Overlay

What you see through the glasses:

```
┌─────────────────────────────────────────┐
│                                    🛒 3 │  ← Cart icon with count
│                                   $24.50│  ← Running total
│                                         │
│                                         │
│          ┌─────────────┐                │
│          │ Cheerios    │ ← Product info │
│          │ $4.99       │   popup when   │
│          │ ✓ Added     │   you grab     │
│          └─────────────┘   something    │
│                                         │
│                                         │
└─────────────────────────────────────────┘
```

## 🚀 Roadmap

### Phase 1: Proof of Concept
- [ ] Basic product recognition (10-20 products)
- [ ] Simple AR overlay showing recognized items
- [ ] Mock cart functionality

### Phase 2: Core Features
- [ ] Hand tracking for pick up/put down detection
- [ ] Real cart with add/remove
- [ ] Barcode scanning fallback
- [ ] Basic companion app

### Phase 3: Store Integration
- [ ] Store product database API
- [ ] Geofencing for store entry/exit
- [ ] Payment integration
- [ ] Receipt generation

### Phase 4: Polish
- [ ] Multi-store support
- [ ] Shopping lists integration
- [ ] Price comparison
- [ ] Purchase history

## 🤔 Challenges to Solve

1. **Product Recognition Accuracy** - Need high confidence before adding to cart
2. **Put-Back Detection** - How to know when item is returned to shelf?
3. **Store Partnerships** - Need product databases from stores
4. **Theft Prevention** - How to prevent "oops I forgot to pay"
5. **Multiple Similar Items** - Grabbing 3 of the same thing

## 📂 Project Structure

```
ARCart/
├── README.md
├── docs/
│   ├── architecture.md
│   └── user-research.md
├── ar-client/          # AR glasses app
│   ├── recognition/    # ML models
│   ├── tracking/       # Hand/object tracking
│   └── ui/             # AR overlays
├── mobile-app/         # Companion app
├── backend/            # API server
│   ├── products/
│   ├── cart/
│   └── payments/
└── ml-training/        # Product recognition training
```

## 🏪 Target Platforms

- **Meta Quest 3** (near-term, good dev support)
- **Apple Vision Pro** (premium experience)
- **Ray-Ban Meta** (everyday glasses form factor)
- **Future lightweight AR glasses**

---

Built with 🛒 by Turbo ⚡
