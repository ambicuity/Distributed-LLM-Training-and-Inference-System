"""
Observability and metrics collection for distributed training and inference.
"""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

import psutil
import torch
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from rich.console import Console

console = Console()

@dataclass
class SystemMetrics:
    """System-level metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_total: List[float] = field(default_factory=list)
    network_sent_mbps: float = 0.0
    network_recv_mbps: float = 0.0
    disk_read_mbps: float = 0.0
    disk_write_mbps: float = 0.0

@dataclass
class TrainingMetrics:
    """Training-specific metrics."""
    loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    step: int = 0
    epoch: int = 0
    flops_per_second: float = 0.0

@dataclass 
class InferenceMetrics:
    """Inference-specific metrics."""
    request_latency: float = 0.0
    tokens_per_second: float = 0.0
    batch_size: int = 0
    queue_length: int = 0
    active_requests: int = 0
    throughput_requests_per_second: float = 0.0
    
class MetricsCollector:
    """Collects and aggregates metrics from various sources."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self._collecting = False
        self._collection_thread: Optional[threading.Thread] = None
        
        # Metrics storage
        self.system_metrics = SystemMetrics()
        self.training_metrics = TrainingMetrics()
        self.inference_metrics = InferenceMetrics()
        
        # Historical data (keep last 1000 points)
        self.system_history: deque = deque(maxlen=1000)
        self.training_history: deque = deque(maxlen=1000)
        self.inference_history: deque = deque(maxlen=1000)
        
        # Network/disk baseline for rate calculation
        self._last_network_stats = None
        self._last_disk_stats = None
        self._last_collection_time = None
        
    def start_collection(self):
        """Start background metrics collection."""
        if self._collecting:
            return
        
        self._collecting = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self._collection_thread.start()
        console.print("[blue]Metrics collection started[/blue]")
    
    def stop_collection(self):
        """Stop background metrics collection."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join()
        console.print("[yellow]Metrics collection stopped[/yellow]")
    
    def _collection_loop(self):
        """Background collection loop."""
        while self._collecting:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                console.print(f"[red]Metrics collection error: {e}[/red]")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        current_time = time.time()
        
        # CPU and memory
        self.system_metrics.cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        self.system_metrics.memory_percent = memory.percent
        self.system_metrics.memory_used_gb = memory.used / (1024**3)
        self.system_metrics.memory_total_gb = memory.total / (1024**3)
        
        # GPU metrics
        if torch.cuda.is_available():
            gpu_utils = []
            gpu_mem_used = []
            gpu_mem_total = []
            
            for i in range(torch.cuda.device_count()):
                # GPU utilization (approximation)
                gpu_utils.append(0.0)  # Would need nvidia-ml-py for real GPU util
                
                # GPU memory
                if torch.cuda.is_initialized():
                    mem_used = torch.cuda.memory_allocated(i) / (1024**3)
                    mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                else:
                    mem_used = 0.0
                    mem_total = 0.0
                
                gpu_mem_used.append(mem_used)
                gpu_mem_total.append(mem_total)
            
            self.system_metrics.gpu_utilization = gpu_utils
            self.system_metrics.gpu_memory_used = gpu_mem_used
            self.system_metrics.gpu_memory_total = gpu_mem_total
        
        # Network I/O
        network_stats = psutil.net_io_counters()
        if self._last_network_stats and self._last_collection_time:
            time_delta = current_time - self._last_collection_time
            sent_delta = network_stats.bytes_sent - self._last_network_stats.bytes_sent
            recv_delta = network_stats.bytes_recv - self._last_network_stats.bytes_recv
            
            self.system_metrics.network_sent_mbps = (sent_delta / time_delta) / (1024**2) * 8
            self.system_metrics.network_recv_mbps = (recv_delta / time_delta) / (1024**2) * 8
        
        self._last_network_stats = network_stats
        
        # Disk I/O  
        disk_stats = psutil.disk_io_counters()
        if disk_stats and self._last_disk_stats and self._last_collection_time:
            time_delta = current_time - self._last_collection_time
            read_delta = disk_stats.read_bytes - self._last_disk_stats.read_bytes
            write_delta = disk_stats.write_bytes - self._last_disk_stats.write_bytes
            
            self.system_metrics.disk_read_mbps = (read_delta / time_delta) / (1024**2)
            self.system_metrics.disk_write_mbps = (write_delta / time_delta) / (1024**2)
        
        if disk_stats:
            self._last_disk_stats = disk_stats
        
        self._last_collection_time = current_time
        
        # Store in history
        self.system_history.append({
            "timestamp": current_time,
            "metrics": self.system_metrics
        })
    
    def update_training_metrics(self, **kwargs):
        """Update training metrics."""
        for key, value in kwargs.items():
            if hasattr(self.training_metrics, key):
                setattr(self.training_metrics, key, value)
        
        self.training_history.append({
            "timestamp": time.time(),
            "metrics": self.training_metrics
        })
    
    def update_inference_metrics(self, **kwargs):
        """Update inference metrics."""
        for key, value in kwargs.items():
            if hasattr(self.inference_metrics, key):
                setattr(self.inference_metrics, key, value)
        
        self.inference_history.append({
            "timestamp": time.time(),
            "metrics": self.inference_metrics
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        return {
            "system": {
                "cpu_percent": self.system_metrics.cpu_percent,
                "memory_percent": self.system_metrics.memory_percent,
                "memory_used_gb": self.system_metrics.memory_used_gb,
                "gpu_memory_used": self.system_metrics.gpu_memory_used,
            },
            "training": {
                "loss": self.training_metrics.loss,
                "learning_rate": self.training_metrics.learning_rate,
                "tokens_per_second": self.training_metrics.tokens_per_second,
                "step": self.training_metrics.step,
            },
            "inference": {
                "request_latency": self.inference_metrics.request_latency,
                "tokens_per_second": self.inference_metrics.tokens_per_second,
                "active_requests": self.inference_metrics.active_requests,
                "throughput_rps": self.inference_metrics.throughput_requests_per_second,
            }
        }

class PrometheusExporter:
    """Exports metrics to Prometheus."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        
        # Training metrics
        self.training_loss = Gauge('llmctl_training_loss', 'Current training loss')
        self.training_lr = Gauge('llmctl_training_learning_rate', 'Current learning rate')
        self.training_step = Counter('llmctl_training_steps_total', 'Total training steps')
        self.training_tokens_per_sec = Gauge('llmctl_training_tokens_per_second', 'Training tokens per second')
        
        # Inference metrics
        self.inference_latency = Histogram('llmctl_inference_latency_seconds', 'Request latency')
        self.inference_requests = Counter('llmctl_inference_requests_total', 'Total inference requests')
        self.inference_active = Gauge('llmctl_inference_active_requests', 'Active inference requests')
        self.inference_throughput = Gauge('llmctl_inference_tokens_per_second', 'Inference tokens per second')
        
        # System metrics
        self.system_cpu = Gauge('llmctl_system_cpu_percent', 'CPU utilization')
        self.system_memory = Gauge('llmctl_system_memory_percent', 'Memory utilization')
        self.system_gpu_memory = Gauge('llmctl_system_gpu_memory_used_gb', 'GPU memory used', ['gpu_id'])
        
    def start_server(self):
        """Start Prometheus metrics server."""
        start_http_server(self.port)
        console.print(f"[green]Prometheus metrics server started on port {self.port}[/green]")
    
    def update_metrics(self, collector: MetricsCollector):
        """Update Prometheus metrics from collector."""
        # Training metrics
        self.training_loss.set(collector.training_metrics.loss)
        self.training_lr.set(collector.training_metrics.learning_rate)
        self.training_tokens_per_sec.set(collector.training_metrics.tokens_per_second)
        
        # Inference metrics
        self.inference_active.set(collector.inference_metrics.active_requests)
        self.inference_throughput.set(collector.inference_metrics.tokens_per_second)
        
        # System metrics
        self.system_cpu.set(collector.system_metrics.cpu_percent)
        self.system_memory.set(collector.system_metrics.memory_percent)
        
        for i, mem_used in enumerate(collector.system_metrics.gpu_memory_used):
            self.system_gpu_memory.labels(gpu_id=str(i)).set(mem_used)

class OpenTelemetryExporter:
    """Exports traces and metrics to OpenTelemetry."""
    
    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or "http://localhost:4318"
        
        # Setup tracing
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        otlp_exporter = OTLPSpanExporter(endpoint=f"{self.endpoint}/v1/traces")
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Setup metrics
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=f"{self.endpoint}/v1/metrics"),
            export_interval_millis=5000,
        )
        metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
        
        self.tracer = tracer
        self.meter = metrics.get_meter(__name__)
        
        # Create instruments
        self.training_loss_histogram = self.meter.create_histogram(
            "llmctl.training.loss",
            description="Training loss values"
        )
        self.inference_latency_histogram = self.meter.create_histogram(
            "llmctl.inference.latency",
            description="Inference request latency"
        )
        
        console.print(f"[green]OpenTelemetry exporter configured for {self.endpoint}[/green]")
    
    def record_training_step(self, loss: float, step: int, **attributes):
        """Record a training step with tracing."""
        with self.tracer.start_as_current_span("training_step") as span:
            span.set_attributes({
                "step": step,
                "loss": loss,
                **attributes
            })
            self.training_loss_histogram.record(loss, {"step": step})
    
    def record_inference_request(self, latency: float, **attributes):
        """Record an inference request with tracing."""
        with self.tracer.start_as_current_span("inference_request") as span:
            span.set_attributes({
                "latency": latency,
                **attributes
            })
            self.inference_latency_histogram.record(latency, attributes)

class ObservabilityManager:
    """Main class for managing observability features."""
    
    def __init__(self, 
                 enable_prometheus: bool = True,
                 prometheus_port: int = 8000,
                 enable_otlp: bool = False,
                 otlp_endpoint: Optional[str] = None,
                 collection_interval: float = 1.0):
        
        self.collector = MetricsCollector(collection_interval)
        
        self.prometheus_exporter = None
        if enable_prometheus:
            self.prometheus_exporter = PrometheusExporter(prometheus_port)
            
        self.otlp_exporter = None
        if enable_otlp:
            self.otlp_exporter = OpenTelemetryExporter(otlp_endpoint)
        
        self._update_thread = None
        self._updating = False
    
    def start(self):
        """Start all observability components."""
        console.print("[blue]Starting observability manager...[/blue]")
        
        # Start metrics collection
        self.collector.start_collection()
        
        # Start Prometheus server
        if self.prometheus_exporter:
            self.prometheus_exporter.start_server()
        
        # Start metrics update loop
        self._updating = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        
        console.print("[green]✓ Observability manager started[/green]")
    
    def stop(self):
        """Stop all observability components."""
        console.print("[yellow]Stopping observability manager...[/yellow]")
        
        self._updating = False
        if self._update_thread:
            self._update_thread.join()
        
        self.collector.stop_collection()
        
        console.print("[yellow]Observability manager stopped[/yellow]")
    
    def _update_loop(self):
        """Update exporters with latest metrics."""
        while self._updating:
            try:
                if self.prometheus_exporter:
                    self.prometheus_exporter.update_metrics(self.collector)
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                console.print(f"[red]Observability update error: {e}[/red]")
                time.sleep(5)
    
    def record_training_step(self, **kwargs):
        """Record training step metrics."""
        self.collector.update_training_metrics(**kwargs)
        
        if self.otlp_exporter and 'loss' in kwargs and 'step' in kwargs:
            self.otlp_exporter.record_training_step(
                kwargs['loss'], 
                kwargs['step'],
                **{k: v for k, v in kwargs.items() if k not in ['loss', 'step']}
            )
    
    def record_inference_request(self, latency: float, **kwargs):
        """Record inference request metrics."""
        self.collector.update_inference_metrics(request_latency=latency, **kwargs)
        
        if self.otlp_exporter:
            self.otlp_exporter.record_inference_request(latency, **kwargs)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        return self.collector.get_summary()

# Global observability manager instance
_global_observability_manager: Optional[ObservabilityManager] = None

def get_observability_manager() -> Optional[ObservabilityManager]:
    """Get the global observability manager instance."""
    return _global_observability_manager

def setup_observability(**kwargs) -> ObservabilityManager:
    """Setup and return global observability manager."""
    global _global_observability_manager
    _global_observability_manager = ObservabilityManager(**kwargs)
    return _global_observability_manager