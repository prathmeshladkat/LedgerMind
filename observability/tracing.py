# observability/tracing.py
# sets up OpenTelemetry distributed tracing
#
# PATTERN: Sidecar (observability)
# our app emits traces to localhost:4317
# the OTel collector container (sidecar) receives them
# and forwards to Prometheus/Grafana
# app code has zero knowledge of where traces go
# you can swap the sidecar config without touching app code
#
# what is a trace?
# a trace = one complete request through the system
# e.g. POST /analyze → fetch → chunk → extract → validate → emit
# each step is a "span" inside the trace
# Grafana shows you the full journey and where time was spent

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from infra.settings import get_settings
import logging

logger = logging.getLogger(__name__)

# module level tracer - import this in other files to create spans
_tracer = None


def init_tracing(service_name: str = "capitalsense"):
    """
    initializes OpenTelemetry tracing.
    call once at app startup with the service name.
    service_name appears in Grafana so you can tell which
    service produced which trace (gateway vs worker vs voice).

    PATTERN: Sidecar
    we send traces to localhost:4317 (OTel collector)
    collector runs as a separate docker container
    app doesn't care what happens to traces after that
    """
    global _tracer

    settings = get_settings()

    # resource identifies this service in traces
    # shows up as service.name in Grafana
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "0.1.0",
        "deployment.environment": settings.environment,
    })

    # OTLP exporter sends traces to OTel collector
    # collector is at localhost:4317 (docker-compose)
    exporter = OTLPSpanExporter(
        endpoint=settings.otel_endpoint,
        insecure=True,   # no TLS for local docker
    )

    # BatchSpanProcessor buffers spans and sends in batches
    # more efficient than sending each span immediately
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # register as global tracer provider
    # now trace.get_tracer() anywhere in app returns this provider
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(service_name)
    logger.info(f"Tracing initialized: service={service_name}")

    return _tracer


def get_tracer() -> trace.Tracer:
    """
    returns the tracer.
    import this wherever you want to create spans.
    if tracing not initialized, returns a no-op tracer
    that does nothing (safe for tests).
    """
    if _tracer is None:
        # return no-op tracer so code works without tracing initialized
        return trace.get_tracer("capitalsense-noop")
    return _tracer


def trace_agent_node(node_name: str):
    """
    decorator to add tracing to any agent node function.
    wraps the node in a span so Grafana shows how long it took.

    usage:
        @trace_agent_node("fetch_filing")
        async def fetch_filing_node(state):
            ...

    this creates a span named "agent.fetch_filing" in every trace.
    you can see in Grafana which agent is the bottleneck.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()

            # create a span for this agent node execution
            with tracer.start_as_current_span(f"agent.{node_name}") as span:
                try:
                    # add state info as span attributes
                    # visible in Grafana trace details
                    if args and isinstance(args[0], dict):
                        state = args[0]
                        span.set_attribute(
                            "agent.ticker",
                            state.get("ticker", "unknown")
                        )
                        span.set_attribute(
                            "agent.filing_type",
                            state.get("filing_type", "unknown")
                        )
                        span.set_attribute(
                            "agent.thread_id",
                            state.get("thread_id", "unknown")
                        )

                    result = await func(*args, **kwargs)

                    # mark span as success
                    span.set_attribute("agent.status", "success")
                    return result

                except Exception as e:
                    # mark span as error with message
                    span.set_attribute("agent.status", "error")
                    span.set_attribute("agent.error", str(e))
                    span.record_exception(e)
                    raise

        # preserve function name for LangGraph node registration
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator