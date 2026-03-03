"""
Flask Web Application for Customer Segmentation + ML Clustering + Churn Prediction
"""

import os
import pickle
import uuid
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    session,
    has_request_context,
)
from werkzeug.utils import secure_filename

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

from segmentation import full_pipeline

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------

app = Flask(__name__)
app.secret_key = os.urandom(32)

UPLOAD_FOLDER = 'uploads'
SESSION_DATA_FOLDER = os.path.join(UPLOAD_FOLDER, 'session_data')
CHURN_MODEL_PATH = 'logistics_regression.pkl'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSION_DATA_FOLDER, exist_ok=True)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_session_id():
    sid = session.get('session_id')
    if not sid:
        sid = uuid.uuid4().hex
        session['session_id'] = sid
    return sid


def _session_state_path():
    return os.path.join(SESSION_DATA_FOLDER, f"{_get_session_id()}.pkl")


def _save_state(state):
    with open(_session_state_path(), 'wb') as f:
        pickle.dump(state, f)


def _load_state():
    path = _session_state_path()
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_churn_model():
    if not os.path.exists(CHURN_MODEL_PATH):
        return None, None
    try:
        with open(CHURN_MODEL_PATH, 'rb') as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            model = payload.get('model')
            scaler = payload.get('scaler')
            return model, scaler
        return None, None
    except Exception:
        return None, None


def _save_churn_model(model, scaler):
    if model is None or scaler is None:
        return
    payload = {'model': model, 'scaler': scaler}
    with open(CHURN_MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)


# --------------------------------------------------
# SEGMENTATION PIPELINE
# --------------------------------------------------

def process_segmentation(df, revenue_col):

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", format="mixed", dayfirst=True)
    df = df.dropna(subset=["InvoiceDate"])

    if revenue_col != "Revenue":
        df["Revenue"] = df[revenue_col]

    rfm = full_pipeline(df)

    return rfm


# --------------------------------------------------
# CHURN MODEL
# --------------------------------------------------

def train_churn_model(rfm_df):

    rfm_df = rfm_df.copy()
    rfm_df['Churn'] = (rfm_df['Recency'] > 180).astype(int)

    feature_cols = ['Recency', 'Frequency', 'Monetary', 'AOV']

    X = rfm_df[feature_cols].fillna(0)
    y = rfm_df['Churn']

    if len(y.unique()) <= 1:
        return np.zeros(len(y)), None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_counts = y.value_counts()

    model = LogisticRegression(max_iter=1000)

    # For small/imbalanced data, train on full set to avoid split failures.
    if len(rfm_df) < 10 or class_counts.min() < 2:
        try:
            model.fit(X_scaled, y)
            probs = model.predict_proba(X_scaled)[:, 1]
            return probs, model, scaler
        except ValueError:
            return np.zeros(len(y)), None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_scaled)[:, 1]

    return probs, model, scaler


def predict_churn(state):

    df = state['df']

    if 'Churn_Probability' not in df.columns:
        feature_cols = ['Recency', 'Frequency', 'Monetary', 'AOV']
        X = df[feature_cols].fillna(0)
        y = (df['Recency'] > 180).astype(int)

        if len(y.unique()) <= 1:
            probs = np.zeros(len(y))
            model, scaler = None, None
        else:
            model, scaler = _load_churn_model()
            if model is not None and scaler is not None:
                try:
                    probs = model.predict_proba(scaler.transform(X))[:, 1]
                except Exception:
                    probs, model, scaler = train_churn_model(df)
                    _save_churn_model(model, scaler)
            else:
                probs, model, scaler = train_churn_model(df)
                _save_churn_model(model, scaler)

        df['Churn_Probability'] = probs

        def risk(p):
            if p >= 0.7:
                return "High Risk"
            elif p >= 0.4:
                return "Medium Risk"
            elif p >= 0.2:
                return "Low Risk"
            return "Safe"

        df['Risk_Level'] = df['Churn_Probability'].apply(risk)

        state['df'] = df

        if has_request_context():
            _save_state(state)

    return state


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            lower_filename = filename.lower()
            if lower_filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            required = ['CustomerID', 'InvoiceDate', 'InvoiceNo']
            for col in required:
                if col not in df.columns:
                    flash(f'Missing column {col}', 'error')
                    return redirect(url_for('index'))

            revenue_col = None
            for col in ['Revenue', 'Total', 'Amount', 'Price']:
                if col in df.columns:
                    revenue_col = col
                    break

            if revenue_col is None:
                flash("Revenue column not found", 'error')
                return redirect(url_for('index'))

            result_df = process_segmentation(df, revenue_col)

            state = {
                'filename': filename,
                'df': result_df
            }

            _save_state(state)

            return redirect(url_for('results'))

        except Exception as e:
            flash(str(e), 'error')
            return redirect(url_for('index'))

    flash('Invalid file', 'error')
    return redirect(url_for('index'))


@app.route('/results')
def results():

    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    state = predict_churn(state)
    df = state['df']

    segment_counts = df['Segment'].value_counts()
    cluster_counts = df['Cluster'].value_counts().sort_index()

    summary = {
        'total_customers': len(df),
        'avg_recency': round(df['Recency'].mean(), 1),
        'avg_frequency': round(df['Frequency'].mean(), 1),
        'avg_monetary': round(df['Monetary'].mean(), 2),
        'avg_churn_prob': round(df['Churn_Probability'].mean() * 100, 1)
    }

    at_risk_customers = df.nlargest(10, 'Churn_Probability').to_dict('records')

    return render_template(
        'results.html',
        filename=state['filename'],
        segment_counts=segment_counts.to_dict(),
        cluster_counts=cluster_counts.to_dict(),
        summary=summary,
        at_risk_customers=at_risk_customers
    )


@app.route('/customers')
def customers():

    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    state = predict_churn(state)
    df = state['df']

    selected_segment = request.args.get('segment', 'All')
    page = request.args.get('page', 1, type=int)
    per_page = 25

    segments = ['All'] + sorted(df['Segment'].dropna().unique().tolist())

    filtered_df = df
    if selected_segment != 'All':
        filtered_df = df[df['Segment'] == selected_segment]

    total = len(filtered_df)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(max(page, 1), total_pages)

    start = (page - 1) * per_page
    end = start + per_page
    page_df = filtered_df.iloc[start:end]

    return render_template(
        'customers.html',
        customers=page_df.to_dict('records'),
        segments=segments,
        current_segment=selected_segment,
        page=page,
        total_pages=total_pages
    )


@app.route('/segment/<segment_name>')
def segment_detail(segment_name):

    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    state = predict_churn(state)
    df = state['df']

    seg_df = df[df['Segment'] == segment_name]
    if seg_df.empty:
        flash(f'Segment "{segment_name}" not found.', 'error')
        return redirect(url_for('results'))

    stats = {
        'count': int(len(seg_df)),
        'avg_recency': round(seg_df['Recency'].mean(), 1),
        'avg_frequency': round(seg_df['Frequency'].mean(), 1),
        'avg_monetary': round(seg_df['Monetary'].mean(), 2),
        'total_revenue': round(seg_df['Monetary'].sum(), 2),
        'avg_churn_prob': round(seg_df['Churn_Probability'].mean() * 100, 1),
    }

    top_customers = seg_df.nlargest(20, 'Monetary').to_dict('records')

    return render_template(
        'segment_detail.html',
        segment_name=segment_name,
        stats=stats,
        top_customers=top_customers
    )


@app.route('/charts')
def charts():

    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    state = predict_churn(state)
    df = state['df']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    seg = df['Segment'].value_counts()
    axes[0].pie(seg.values, labels=seg.index, autopct='%1.1f%%')
    axes[0].set_title("RFM Segments")

    clusters = df['Cluster'].value_counts().sort_index()
    axes[1].bar(clusters.index.astype(str), clusters.values)
    axes[1].set_title("ML Cluster Distribution")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Customers")

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()

    plt.close()

    return render_template("charts.html", chart_url=chart_url)


@app.route('/download')
def download():

    state = _load_state()
    if state is None:
        return redirect(url_for('index'))

    state = predict_churn(state)
    df = state['df']

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='SegmentationResults')
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name='segmentation_results.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.route('/reset')
def reset():
    path = _session_state_path()
    if os.path.exists(path):
        os.remove(path)
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
