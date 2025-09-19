"""
Health monitoring and fault tolerance for distributed LLM systems.
"""

import time
import threading
import psutil
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
from rich.console import Console

console = Console()

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical" 
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_function: Callable[[], bool]
    critical: bool = False
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    description: str = ""

@dataclass
class HealthReport:
    """Health report for a component."""
    component: str
    status: HealthStatus
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    
class SystemHealthMonitor:
    """Monitors system-level health metrics."""
    
    def __init__(self):
        self.checks = [
            HealthCheck(
                name="cpu_usage",
                check_function=self._check_cpu_usage,
                warning_threshold=80.0,
                critical_threshold=95.0,
                description="CPU utilization percentage"
            ),
            HealthCheck(
                name="memory_usage", 
                check_function=self._check_memory_usage,
                warning_threshold=85.0,
                critical_threshold=95.0,
                description="Memory utilization percentage"
            ),
            HealthCheck(
                name="disk_space",
                check_function=self._check_disk_space,
                warning_threshold=85.0,
                critical_threshold=95.0,
                description="Disk space utilization percentage"
            ),
            HealthCheck(
                name="gpu_memory",
                check_function=self._check_gpu_memory,
                warning_threshold=90.0,
                critical_threshold=98.0,
                description="GPU memory utilization percentage"
            )
        ]
        
        self.last_metrics = {}
    
    def _check_cpu_usage(self) -> bool:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        self.last_metrics["cpu_usage"] = cpu_percent
        return cpu_percent < 95.0
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        self.last_metrics["memory_usage"] = memory.percent
        return memory.percent < 95.0
    
    def _check_disk_space(self) -> bool:
        """Check disk space."""
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.last_metrics["disk_usage"] = disk_percent
        return disk_percent < 95.0
    
    def _check_gpu_memory(self) -> bool:
        """Check GPU memory usage."""
        if not torch.cuda.is_available():
            return True
        
        try:
            max_usage = 0
            for i in range(torch.cuda.device_count()):
                if torch.cuda.is_initialized():
                    used = torch.cuda.memory_allocated(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    usage_percent = (used / total) * 100
                    max_usage = max(max_usage, usage_percent)
            
            self.last_metrics["gpu_memory_usage"] = max_usage
            return max_usage < 98.0
        except Exception:
            return True
    
    def get_health_report(self) -> HealthReport:
        """Generate system health report."""
        checks_passed = {}
        overall_status = HealthStatus.HEALTHY
        messages = []
        
        for check in self.checks:
            try:
                passed = check.check_function()
                checks_passed[check.name] = passed
                
                # Check metric against thresholds
                metric_value = self.last_metrics.get(check.name, 0)
                
                if check.critical_threshold and metric_value >= check.critical_threshold:
                    overall_status = HealthStatus.CRITICAL
                    messages.append(f"{check.name} critical: {metric_value:.1f}%")
                elif check.warning_threshold and metric_value >= check.warning_threshold:
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.WARNING
                    messages.append(f"{check.name} warning: {metric_value:.1f}%")
                
            except Exception as e:
                checks_passed[check.name] = False
                overall_status = HealthStatus.CRITICAL
                messages.append(f"{check.name} check failed: {e}")
        
        return HealthReport(
            component="system",
            status=overall_status,
            checks=checks_passed,
            metrics=self.last_metrics.copy(),
            message="; ".join(messages) if messages else "All systems normal"
        )

class TrainingHealthMonitor:
    """Monitors training-specific health metrics."""
    
    def __init__(self):
        self.training_metrics = {}
        self.last_update = time.time()
        
    def update_training_metrics(self, **kwargs):
        """Update training metrics."""
        self.training_metrics.update(kwargs)
        self.last_update = time.time()
    
    def get_health_report(self) -> HealthReport:
        """Generate training health report."""
        current_time = time.time()
        time_since_update = current_time - self.last_update
        
        status = HealthStatus.HEALTHY
        checks = {}
        messages = []
        
        # Check if training is active (metrics updated recently)
        checks["training_active"] = time_since_update < 300  # 5 minutes
        if not checks["training_active"]:
            status = HealthStatus.WARNING
            messages.append(f"No training updates for {time_since_update:.0f}s")
        
        # Check loss trends
        if "loss" in self.training_metrics:
            loss = self.training_metrics["loss"]
            checks["loss_finite"] = not (torch.isinf(torch.tensor(loss)) or torch.isnan(torch.tensor(loss)))
            if not checks["loss_finite"]:
                status = HealthStatus.CRITICAL
                messages.append("Loss is infinite or NaN")
        else:
            checks["loss_finite"] = False
        
        # Check gradient health
        if "gradient_norm" in self.training_metrics:
            grad_norm = self.training_metrics["gradient_norm"]
            checks["gradient_healthy"] = 0.001 < grad_norm < 100.0
            if not checks["gradient_healthy"]:
                if status != HealthStatus.CRITICAL:
                    status = HealthStatus.WARNING
                messages.append(f"Unusual gradient norm: {grad_norm:.4f}")
        else:
            checks["gradient_healthy"] = True
            
        return HealthReport(
            component="training",
            status=status,
            checks=checks,
            metrics=self.training_metrics.copy(),
            message="; ".join(messages) if messages else "Training healthy"
        )

class InferenceHealthMonitor:
    """Monitors inference-specific health metrics."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.active_requests = 0
        self.last_request_time = time.time()
        
    def record_request(self, latency: float, success: bool = True):
        """Record an inference request."""
        self.request_count += 1
        if not success:
            self.error_count += 1
        self.total_latency += latency
        self.last_request_time = time.time()
    
    def update_active_requests(self, count: int):
        """Update active request count."""
        self.active_requests = count
    
    def get_health_report(self) -> HealthReport:
        """Generate inference health report."""
        current_time = time.time()
        status = HealthStatus.HEALTHY
        checks = {}
        messages = []
        metrics = {}
        
        # Calculate error rate
        error_rate = (self.error_count / max(self.request_count, 1)) * 100
        checks["low_error_rate"] = error_rate < 5.0
        if not checks["low_error_rate"]:
            status = HealthStatus.WARNING
            messages.append(f"High error rate: {error_rate:.1f}%")
        
        # Calculate average latency
        avg_latency = self.total_latency / max(self.request_count, 1)
        checks["reasonable_latency"] = avg_latency < 10.0  # 10 seconds
        if not checks["reasonable_latency"]:
            if status != HealthStatus.CRITICAL:
                status = HealthStatus.WARNING
            messages.append(f"High latency: {avg_latency:.2f}s")
        
        # Check if requests are being processed
        time_since_request = current_time - self.last_request_time
        checks["recent_activity"] = time_since_request < 600  # 10 minutes
        
        # Check for request queue buildup
        checks["queue_manageable"] = self.active_requests < 100
        if not checks["queue_manageable"]:
            status = HealthStatus.CRITICAL
            messages.append(f"Request queue overloaded: {self.active_requests}")
        
        metrics.update({
            "request_count": self.request_count,
            "error_rate": error_rate,
            "avg_latency": avg_latency,
            "active_requests": self.active_requests,
        })
        
        return HealthReport(
            component="inference",
            status=status,
            checks=checks,
            metrics=metrics,
            message="; ".join(messages) if messages else "Inference healthy"
        )

class HealthManager:
    """Central health monitoring and management system."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.monitors = {
            "system": SystemHealthMonitor(),
            "training": TrainingHealthMonitor(),
            "inference": InferenceHealthMonitor()
        }
        
        self.health_history: List[Dict[str, HealthReport]] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[HealthReport], None]] = []
        
    def add_alert_callback(self, callback: Callable[[HealthReport], None]):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        console.print("[blue]Health monitoring started[/blue]")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        console.print("[yellow]Health monitoring stopped[/yellow]")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                console.print(f"[red]Health monitoring error: {e}[/red]")
                time.sleep(self.check_interval)
    
    def _perform_health_checks(self):
        """Perform health checks and generate reports."""
        reports = {}
        
        for name, monitor in self.monitors.items():
            try:
                report = monitor.get_health_report()
                reports[name] = report
                
                # Trigger alerts for warnings and critical issues
                if report.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                    for callback in self.alert_callbacks:
                        try:
                            callback(report)
                        except Exception as e:
                            console.print(f"[red]Alert callback error: {e}[/red]")
                            
            except Exception as e:
                console.print(f"[red]Health check error for {name}: {e}[/red]")
        
        # Store in history
        self.health_history.append(reports)
        if len(self.health_history) > 1000:  # Keep last 1000 reports
            self.health_history.pop(0)
    
    def get_current_health(self) -> Dict[str, HealthReport]:
        """Get current health status."""
        reports = {}
        for name, monitor in self.monitors.items():
            reports[name] = monitor.get_health_report()
        return reports
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        current_health = self.get_current_health()
        
        statuses = [report.status for report in current_health.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def update_training_metrics(self, **kwargs):
        """Update training metrics."""
        if "training" in self.monitors:
            self.monitors["training"].update_training_metrics(**kwargs)
    
    def record_inference_request(self, latency: float, success: bool = True):
        """Record an inference request."""
        if "inference" in self.monitors:
            self.monitors["inference"].record_request(latency, success)
    
    def update_active_requests(self, count: int):
        """Update active request count."""
        if "inference" in self.monitors:
            self.monitors["inference"].update_active_requests(count)
    
    def save_health_report(self, filepath: str):
        """Save health report to file."""
        current_health = self.get_current_health()
        
        # Convert to serializable format
        report_data = {}
        for name, report in current_health.items():
            report_data[name] = {
                "component": report.component,
                "status": report.status.value,
                "checks": report.checks,
                "metrics": report.metrics,
                "message": report.message,
                "timestamp": report.timestamp
            }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        console.print(f"[green]Health report saved to {filepath}[/green]")

def create_default_alert_callback() -> Callable[[HealthReport], None]:
    """Create default alert callback that prints to console."""
    def alert_callback(report: HealthReport):
        if report.status == HealthStatus.CRITICAL:
            console.print(f"[red]🚨 CRITICAL: {report.component} - {report.message}[/red]")
        elif report.status == HealthStatus.WARNING:
            console.print(f"[yellow]⚠️  WARNING: {report.component} - {report.message}[/yellow]")
    
    return alert_callback

# Global health manager instance
_global_health_manager: Optional[HealthManager] = None

def get_health_manager() -> Optional[HealthManager]:
    """Get the global health manager instance."""
    return _global_health_manager

def setup_health_monitoring(**kwargs) -> HealthManager:
    """Setup and return global health manager."""
    global _global_health_manager
    _global_health_manager = HealthManager(**kwargs)
    
    # Add default alert callback
    _global_health_manager.add_alert_callback(create_default_alert_callback())
    
    return _global_health_manager