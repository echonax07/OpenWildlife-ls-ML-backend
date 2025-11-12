import hmac
import logging
import os
import json

from flask import Flask, request, jsonify, Response
from rq.job import Job
from redis import Redis

from .response import ModelResponse
from .model import LabelStudioMLBase
from .exceptions import exception_handler



redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
# ic(redis_host, redis_port)
# Initialize Redis connection
redis_conn = Redis(host=redis_host, port=redis_port)

logger = logging.getLogger(__name__)

_server = Flask(__name__)
MODEL_CLASS = LabelStudioMLBase
BASIC_AUTH = None
from icecream import ic

def init_app(model_class, basic_auth_user=None, basic_auth_pass=None):
    global MODEL_CLASS
    global BASIC_AUTH

    if not issubclass(model_class, LabelStudioMLBase):
        raise ValueError('Inference class should be the subclass of ' + LabelStudioMLBase.__class__.__name__)

    MODEL_CLASS = model_class
    basic_auth_user = basic_auth_user or os.environ.get('BASIC_AUTH_USER')
    basic_auth_pass = basic_auth_pass or os.environ.get('BASIC_AUTH_PASS')
    if basic_auth_user and basic_auth_pass:
        BASIC_AUTH = (basic_auth_user, basic_auth_pass)

    return _server

@_server.route('/queue_predict', methods=['POST'])
@exception_handler
def queue_predict():
    ic("Queued predict is called")
    data = request.json
    tasks = data.get('tasks')
    label_config = data.get('label_config')
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    params = data.get('params', {})
    params["project_id"] = project_id
    context = params.pop('context', {})

    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)

    # model.use_label_config(label_config)

    result = model.queue_predict(tasks, context=context, **params)
    
    if result['error'] is None:
        return jsonify({
            'status': 'Prediction triggered',
            'job': result['job_id']
        }), 200
    else:
        return jsonify({
            'status': 'Prediction setup failed',
            'error': result['error']
        }), 500  # 500 = Internal Server Error



@_server.route('/predict', methods=['POST'])
@exception_handler
def _predict():
    """
    Predict tasks

    Example request:
    request = {
            'tasks': tasks,
            'model_version': model_version,
            'project': '{project.id}.{int(project.created_at.timestamp())}',
            'label_config': project.label_config,
            'params': {
                'login': project.task_data_login,
                'password': project.task_data_password,
                'context': context,
            },
        }

    @return:
    Predictions in LS format
    """
    ic("predict is called")
    data = request.json
    tasks = data.get('tasks')
    label_config = data.get('label_config')
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    params = data.get('params', {})
    params["project_id"] = project_id
    context = params.pop('context', {})

    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)

    # model.use_label_config(label_config)

    response = model.predict(tasks, context=context, **params)

    # if there is no model version we will take the default
    if isinstance(response, ModelResponse):
        if not response.has_model_version():
            mv = model.model_version
            if mv:
                response.set_version(str(mv))
        else:
            response.update_predictions_version()

        response = response.model_dump()

    res = response
    if res is None:
        res = []

    if isinstance(res, dict):
        res = response.get("predictions", response)

    return jsonify({'results': res})


@_server.route('/setup', methods=['POST'])
@exception_handler
def _setup():
    ic("Setup is called")
    data = request.json
    project_id = data.get('project').split('.', 1)[0]
    label_config = data.get('schema')
    extra_params = data.get('extra_params')
    ic(extra_params)
    # If extra_params is a dictionary, convert it to a JSON string
    if isinstance(extra_params, dict):
        extra_params = json.dumps(extra_params)

    model = MODEL_CLASS(project_id=project_id,
                        label_config=label_config)

    if extra_params:
        model.set_extra_params(extra_params)

    model_version = model.get('model_version')
    return jsonify({'model_version': model_version})

TRAIN_EVENTS = (
    'ANNOTATION_CREATED',
    'ANNOTATION_UPDATED',
    'ANNOTATION_DELETED',
    'START_TRAINING'
)

@_server.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    event = data.pop('action')
    if event not in TRAIN_EVENTS:
        # TODO: This is a hard-codey way to handle the PROJECT_UPDATED event specifically for model version updates.
        # Ideally, this should be handled in a more generic way.
        if event == "PROJECT_UPDATED":
            if "model_save_name" in data['project'] and data["project"]["model_save_name"]:
                model = MODEL_CLASS(project_id=str(data['project']['id']),
                                    label_config=data['project']['label_config'])
                model.save_current_version_as(data['project']['model_save_name'])
            if "model_version" in data['project'] and data['project']['model_version']:
                model = MODEL_CLASS(project_id=str(data['project']['id']),
                                    label_config=data['project']['label_config'])
                model.set('model_version', data['project']['model_version'])
        return jsonify({'status': 'Unknown event'}), 200
    project_id = str(data['project']['id'])
    label_config = data['project']['label_config']
    model = MODEL_CLASS(project_id, label_config=label_config)
    result = model.fit(event, data)

    try:
        response = jsonify({'result': result, 'status': 'ok'})
    except Exception as e:
        response = jsonify({'error': str(e), 'status': 'error'})

    return response, 201

@_server.route('/force_train', methods=['POST'])
@exception_handler
def _force_train():
    """
    Force retrain the model on specified tasks.

    Example request:
    {
        'tasks': tasks,           # List of tasks to train on
        'project': project_info,  # Project identifier (e.g., 'project.id.timestamp')
        'label_config': config,   # Label configuration from the project
        'params': { ... }         # Additional parameters (optional)
    }
    """
    data = request.json
    tasks = data.get('tasks')
    project = data.get('project')
    label_config = data.get('label_config')
    params = data.get('params', {})
    from icecream import ic
    ic(data['project'])
    ic(str(data['project']))
    project_id = str(data['project'])

    # Initialize the model with project settings
    model = MODEL_CLASS(project_id=project_id, label_config=label_config)

    # Prepare data structure similar to webhook events
    fit_data = {
        'project': {
            'id': project_id,
            'label_config': label_config
        },
        'tasks': tasks,
        'params': params
    }

    # Trigger model training with a custom event
    result = model.force_fit('FORCE_TRAIN', fit_data)

    if result['error'] is None:
        return jsonify({
            'status': 'Training triggered',
            'job': result['job_id']
        }), 200
    else:
        return jsonify({
            'status': 'Training setup failed',
            'error': result['error']
        }), 500  # 500 = Internal Server Error


@_server.route('/clear_memory_bank', methods=['POST'])
@exception_handler
def _clear_memory_bank():
    data = request.json
    project_id = data.get('project').split('.', 1)[0]
    label_config = data.get('label_config')  # Get label_config from request if needed
    model = MODEL_CLASS(project_id=project_id, label_config=label_config)

    status = model.clear_memory_bank()
    if status:
        return jsonify({
            'status': 'success',
            'message': 'Memory bank cleared successfully',
            'project_id': project_id
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to clear memory bank',
            'project_id': project_id
        }), 500  # 500 = Internal Server Error
        

@_server.route('/health', methods=['GET'])
@_server.route('/', methods=['GET'])
@exception_handler
def health():
    ic("Health is called")
    return jsonify({
        'status': 'UP',
        'model_class': MODEL_CLASS.__name__
    })

@_server.route('/versions', methods=['GET'])
@exception_handler
def versions():
    data = request.json
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    model = MODEL_CLASS(project_id=project_id, label_config=None)
    model_versions = model.get_versions()
    return jsonify({
        'versions': model_versions
    })

@_server.route('/job_status', methods=['GET'])
@exception_handler
def job_status():
    data = request.json
    job_id = data.get('job_id')

    if not job_id:
        return jsonify({
            'message': 'Job ID is required'
        }), 400
    
    job = Job.fetch(job_id, connection=redis_conn)
    if job is None:
        # Do something, return an error.
        return jsonify({
            'message': f'Job with ID {job_id} not found'
        }), 404

    status = job.get_status()
    print(f"JOB STATUS FOR {job_id}: ", status)
    # "scheduled", "deferred" currently not supported.
    if status not in ['queued', 'started', 'finished', 'failed', 'canceled', 'stopped']:
        return jsonify({
            'message': f'Unknown job status: {status}'
        }), 500
    
    if status == 'finished':
        return jsonify({
            'job_status': status,
            'job_id': job_id,
            'result': job.result
        }), 200
    else:
        # API status is 200 OK even if the job fails. We're just returning what it is.
        return jsonify({
            'job_status': status,
            'job_id': job_id,
        }), 200

@_server.route('/custom_weights_path', methods=['POST'])
@exception_handler
def custom_weights_path():
    data = request.json
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    model = MODEL_CLASS(project_id=project_id, label_config=None)
    print(data)
    print("CHECKING WEIGHTS PATH", data.get('custom_weights_path'))
    if model.load_weights_from_path(data.get('custom_weights_path')):
        return jsonify({
            'status': 'success',
            'message': 'Weights loaded successfully'
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to load weights: path not found or invalid'
        }), 400

@_server.route('/extra-params', methods=['GET'])
@exception_handler
def extra_params_config():
    data = request.json
    project = str(data.get('project'))
    project_id = project.split('.', 1)[0] if project else None
    model = MODEL_CLASS(project_id=project_id, label_config=None)
    model_params = model.get_model_extra_params_config()
    ic(model_params)
    return jsonify({
        'extra_params': model_params
    })

@_server.route('/metrics', methods=['GET'])
@exception_handler
def metrics():
    return jsonify({})


@_server.errorhandler(FileNotFoundError)
def file_not_found_error_handler(error):
    logger.warning('Got error: ' + str(error))
    return str(error), 404


@_server.errorhandler(AssertionError)
def assertion_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


@_server.errorhandler(IndexError)
def index_error(error):
    logger.error(str(error), exc_info=True)
    return str(error), 500


def safe_str_cmp(a, b):
    return hmac.compare_digest(a, b)


@_server.before_request
def check_auth():
    if BASIC_AUTH is not None:

        auth = request.authorization
        if not auth or not (safe_str_cmp(auth.username, BASIC_AUTH[0]) and safe_str_cmp(auth.password, BASIC_AUTH[1])):
            return Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})


@_server.before_request
def log_request_info():
    logger.debug('Request headers: %s', request.headers)
    logger.debug('Request body: %s', request.get_data())


@_server.after_request
def log_response_info(response):
    logger.debug('Response status: %s', response.status)
    logger.debug('Response headers: %s', response.headers)
    logger.debug('Response body: %s', response.get_data())
    return response
