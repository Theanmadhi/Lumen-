import { Router } from "express";
import {
  getAllPlans,
  getPlanDetails,
  subscribeToPlan,
  upgradePlan,
  downgradePlan,
  cancelSubscription,
  renewSubscription,
  getSubscriptionHistory,
  getActiveSubscription,
} from "../controllers/userPlanController.js";
// import { protect } from "../middlewares/authMiddleware.js";

const router = Router();

router.route("/plans").get(getAllPlans);
router.route("/plans/:planId").get(getPlanDetails);

// router.use(protect); 

// Subscription management routes
router.route("/subscribe").post(subscribeToPlan);
router.route("/upgrade").post(upgradePlan);
router.route("/downgrade").post(downgradePlan);
router.route("/cancel").post(cancelSubscription);
router.route("/renew").post(renewSubscription);
router.route("/history").get(getSubscriptionHistory);
router.route("/active").get(getActiveSubscription);

export default router;