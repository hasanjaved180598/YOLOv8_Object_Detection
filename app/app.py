
import os
import sys
import json
import requests
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import io

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

API_BASE_URL = 'http://localhost:8000'

st.set_page_config(
    page_title='YOLOv8 Object Detection',
    page_icon='🎯',
    layout='wide'
)


def check_api() -> bool:
    try:
        r = requests.get(f'{API_BASE_URL}/health', timeout=5)
        return r.status_code == 200
    except:
        return False


def detect_objects(image_bytes: bytes, confidence: float) -> dict:
    try:
        r = requests.post(
            f'{API_BASE_URL}/detect',
            files={'file': ('image.jpg', image_bytes, 'image/jpeg')},
            params={'confidence': confidence},
            timeout=30
        )
        r.raise_for_status()
        return {'data': r.json(), 'error': None}
    except requests.exceptions.ConnectionError:
        return {'data': None, 'error': 'Cannot connect to API. Run: python api/main.py'}
    except Exception as e:
        return {'data': None, 'error': str(e)}


def get_annotated_image(image_bytes: bytes, confidence: float) -> dict:
    try:
        r = requests.post(
            f'{API_BASE_URL}/detect/annotated',
            files={'file': ('image.jpg', image_bytes, 'image/jpeg')},
            params={'confidence': confidence},
            timeout=30
        )
        r.raise_for_status()
        count   = int(r.headers.get('X-Detection-Count', 0))
        summary = json.loads(r.headers.get('X-Class-Summary', '{}'))
        image   = Image.open(io.BytesIO(r.content))
        return {'image': image, 'count': count, 'summary': summary, 'error': None}
    except requests.exceptions.ConnectionError:
        return {'image': None, 'count': 0, 'summary': {}, 'error': 'Cannot connect to API.'}
    except Exception as e:
        return {'image': None, 'count': 0, 'summary': {}, 'error': str(e)}


# ── Header ───────────────────────────────────────────────────────────────────
st.title('🎯 YOLOv8 Object Detection')
st.markdown(
    'Powered by **YOLOv8 Nano** pre-trained on COCO dataset. '
    'Upload an image to detect objects instantly. '
    'Detects **80 object classes** including people, vehicles, animals, and everyday items.'
)

if check_api():
    st.success('✅ FastAPI server running at localhost:8000')
else:
    st.error('❌ FastAPI server not running. Start it with: `python api/main.py`')

st.divider()

with st.sidebar:
    st.header('⚙️ Settings')

    confidence = st.slider(
        'Confidence Threshold',
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help='Only show detections above this confidence level'
    )

    st.markdown(f'**Current:** {confidence:.0%} minimum confidence')

    st.divider()
    st.header('📋 Detectable Classes')
    st.markdown("""
    **People & Body Parts**
    person

    **Vehicles**
    car, truck, bus, motorcycle, bicycle, airplane, boat, train

    **Animals**
    dog, cat, bird, horse, cow, sheep, elephant, bear, zebra, giraffe

    **Electronics**
    laptop, phone, keyboard, mouse, remote, TV, microwave

    **Furniture**
    chair, couch, bed, dining table, toilet

    **Food & Kitchen**
    bottle, cup, fork, knife, spoon, bowl, banana, apple, pizza

    *...and more — 80 total classes*
    """)

    st.divider()
    st.header('🔌 API Endpoints')
    st.markdown("""
    | Endpoint | Description |
    |----------|-------------|
    | `GET /health` | Health check |
    | `GET /classes` | List all classes |
    | `POST /detect` | JSON detections |
    | `POST /detect/annotated` | Annotated image |
    | `GET /docs` | Swagger UI |
    """)

uploaded = st.file_uploader(
    'Upload an image',
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    help='Upload any image — works best with clear, well-lit photos'
)

if uploaded:
    image_bytes = uploaded.read()
    original    = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Original Image')
        st.image(original, use_column_width=True)
        st.caption(f'Size: {original.width} × {original.height} px')

    with col2:
        st.subheader('Detection Result')
        with st.spinner('Running YOLOv8 detection ...'):
            result = get_annotated_image(image_bytes, confidence)

        if result['error']:
            st.error(result['error'])
        else:
            st.image(result['image'], use_column_width=True)
            st.caption(f"Detected {result['count']} object(s)")

    st.divider()

    if not result['error']:
        st.subheader('📊 Detection Summary')

        m1, m2, m3 = st.columns(3)
        m1.metric('Objects Detected', result['count'])
        m2.metric('Unique Classes',   len(result['summary']))
        m3.metric('Confidence Min',   f'{confidence:.0%}')

        if result['summary']:
            classes = list(result['summary'].keys())
            counts  = list(result['summary'].values())

            fig = go.Figure(go.Bar(
                x=classes,
                y=counts,
                marker_color='steelblue',
                text=counts,
                textposition='auto'
            ))
            fig.update_layout(
                title='Detected Objects by Class',
                xaxis_title='Class',
                yaxis_title='Count',
                height=350,
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.spinner('Getting detailed detections ...'):
            json_result = detect_objects(image_bytes, confidence)

        if json_result['data'] and json_result['data']['detections']:
            st.subheader('📋 Detailed Detections')

            detections = json_result['data']['detections']
            detections.sort(key=lambda x: x['confidence'], reverse=True)

            for i, det in enumerate(detections):
                with st.expander(
                    f"{i+1}. {det['class'].upper()} — {det['confidence']*100:.1f}% confidence"
                ):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Class:** {det['class']}")
                        st.markdown(f"**Confidence:** {det['confidence']*100:.2f}%")
                    with c2:
                        bbox = det['bbox']
                        st.markdown(f"**Bounding Box:**")
                        st.markdown(
                            f"Top-left: ({bbox['x1']}, {bbox['y1']})  |  "
                            f"Bottom-right: ({bbox['x2']}, {bbox['y2']})"
                        )
                        w = bbox['x2'] - bbox['x1']
                        h = bbox['y2'] - bbox['y1']
                        st.markdown(f"**Size:** {w} × {h} px")

else:
    st.info('👆 Upload an image to start detecting objects.')

    st.subheader('💡 What can this detect?')
    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        st.markdown("""
        **🏙️ Street Scenes**
        - People walking
        - Cars, buses, trucks
        - Traffic lights
        - Stop signs
        """)
    with ex2:
        st.markdown("""
        **🏠 Indoor Scenes**
        - Furniture (chairs, tables)
        - Electronics (laptop, phone)
        - Kitchen items
        - People
        """)
    with ex3:
        st.markdown("""
        **🐾 Nature & Animals**
        - Dogs, cats, birds
        - Horses, cows, sheep
        - Bears, elephants
        - Zebras, giraffes
        """)

    st.divider()
    st.subheader('🔌 Use the API Directly')
    st.code("""
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f},
        params={'confidence': 0.25}
    )

result = response.json()
print(f"Found {result['detection_count']} objects")
for obj in result['detections']:
    print(f"  {obj['class']}: {obj['confidence']*100:.1f}%")
    """, language='python')
