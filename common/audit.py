# -*- coding: utf-8 -*-
"""
Audit Reporting Module
======================

Generates post-execution audit reports for compliance and tracking.
Deleting previous reports ensures only the latest run is tracked.
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Copyright info
COPYRIGHT_HEADER = """
==============================================================================
Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
Email: hkphimanshukumar321@gmail.com
==============================================================================
"""

def generate_audit_report(
    results: Dict[str, Any],
    output_dir: Path,
    report_name: str = "AUDIT_REPORT.md"
) -> Path:
    """
    Generate a fresh audit report, deleting any previous one.
    
    Args:
        results: Dictionary containing 'passed', 'failed', 'total', 'details'
        output_dir: Directory to save the report
        report_name: Filename of the report
        
    Returns:
        Path to the generated report
    """
    output_dir = Path(output_dir)
    report_path = output_dir / report_name
    
    # 1. Delete previous report
    if report_path.exists():
        try:
            os.remove(report_path)
            logger.info(f"Deleted previous audit report: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to delete previous audit: {e}")
            
    # 2. Generate content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    status_icon = "✅" if results.get('failed', 0) == 0 else "❌"
    
    content = f"""# {status_icon} Execution Audit Report
{COPYRIGHT_HEADER}

**Generated**: {timestamp}
**Status**: {'PASSED' if results.get('failed', 0) == 0 else 'FAILED'}

## Summary
| Metric | Count |
|--------|-------|
| Total Tests | {results.get('total', 0)} |
| Passed | {results.get('passed', 0)} |
| Failed | {results.get('failed', 0)} |
| Duration | {results.get('duration', 0):.2f}s |

## Host Information
- **System**: {results.get('system', 'Unknown')}
- **Processor**: {results.get('processor', 'Unknown')}
- **User**: {os.environ.get('USERNAME', 'Unknown')}

## Execution Details
"""
    
    if 'details' in results:
        content += "\n```text\n"
        for item in results['details']:
            content += f"{item}\n"
        content += "```\n"
        
    content += "\n---\n*Verified by Automated Audit System v1.0*"
    
    # 3. Write file
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Generated new audit report: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Failed to write audit report: {e}")
        return None
