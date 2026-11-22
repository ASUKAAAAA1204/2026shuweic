"""Public dispatch entry for the three frozen nominal cases."""
from pathlib import Path
import sys

TIER2 = Path(__file__).resolve().parents[2] / "第三问分工况晋级" / "续轮优化"
TIER4 = TIER2 / "S3三级" / "S3四级"
sys.path.insert(0, str(TIER4))
from deploy import dispatcher
