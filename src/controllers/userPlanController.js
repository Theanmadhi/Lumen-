import { Subscription } from "../models/Subscription.model.js";
import Plan from "../models/planModel.js";
import Billing from "../models/billingModel.js";
import mongoose from "mongoose";
import { createLog } from "./logController.js";

/**
 * Get all available plans for a user to browse
 */
export const getAllPlans = async (req, res) => {
  try {
    const plans = await Plan.find({ status: "active" })
      .select("name description price features tier billingCycle trialPeriod")
      .sort({ price: 1 });

    res.status(200).json({
      success: true,
      data: plans,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "Failed to fetch available plans",
      error: error.message,
    });
  }
};

/**
 * Get details of a specific plan
 */
export const getPlanDetails = async (req, res) => {
  try {
    const { planId } = req.params;

    const plan = await Plan.findById(planId)
      .populate("upgradeTo", "name price features tier")
      .populate("downgradeTo", "name price features tier");

    if (!plan) {
      return res.status(404).json({
        success: false,
        message: "Plan not found",
      });
    }

    res.status(200).json({
      success: true,
      data: plan,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "Failed to fetch plan details",
      error: error.message,
    });
  }
};

/**
 * Subscribe a user to a new plan
 */
export const subscribeToPlan = async (req, res) => {
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const { planId, paymentMethod } = req.body;
    const userId = req.user._id; 

    // Check if plan exists
    const plan = await Plan.findById(planId);
    if (!plan || plan.status !== "active") {
      await session.abortTransaction();
      session.endSession();
      return res.status(404).json({
        success: false,
        message: "Plan not found or not available",
      });
    }

    // Check if user already has an active subscription
    const existingSubscription = await Subscription.findOne({
      user: userId,
      status: "active",
    });

    if (existingSubscription) {
      await session.abortTransaction();
      session.endSession();
      return res.status(400).json({
        success: false,
        message: "You already have an active subscription. Please cancel or upgrade it instead.",
      });
    }

    // Calculate end date based on billing cycle
    const startDate = new Date();
    const endDate = new Date(startDate);

    switch (plan.billingCycle) {
      case "monthly":
        endDate.setMonth(endDate.getMonth() + 1);
        break;
      case "quarterly":
        endDate.setMonth(endDate.getMonth() + 3);
        break;
      case "semi-annual":
        endDate.setMonth(endDate.getMonth() + 6);
        break;
      case "annual":
        endDate.setFullYear(endDate.getFullYear() + 1);
        break;
      default:
        endDate.setMonth(endDate.getMonth() + 1); // Default to monthly
    }

    // Create new subscription
    const newSubscription = await Subscription.create(
      [{
        user: userId,
        plan: planId,
        startDate,
        endDate,
        status: "active",
      }],
      { session }
    );

    // Create billing record
    const billing = await Billing.create(
      [{
        user: userId,
        subscription: newSubscription[0]._id,
        amount: plan.price,
        subtotal: plan.price,
        dueDate: new Date(), 
        billingPeriod: {
          start: startDate,
          end: endDate,
        },
        paymentMethod: {
          type: paymentMethod.type,
          details: paymentMethod.details,
        },
        items: [
          {
            description: `Subscription to ${plan.name} plan`,
            amount: plan.price,
            quantity: 1,
          },
        ],
      }],
      { session }
    );

    // Log the subscription action
    await createLog({
      user: userId,
      action: "SUBSCRIBE",
      entityType: "SUBSCRIPTION",
      entityId: newSubscription[0]._id,
      details: {
        planId: planId,
        planName: plan.name,
        startDate,
        endDate,
      },
    }, session);

    await session.commitTransaction();
    session.endSession();

    res.status(201).json({
      success: true,
      message: "Successfully subscribed to plan",
      data: {
        subscription: newSubscription[0],
        billing: billing[0],
      },
    });
  } catch (error) {
    await session.abortTransaction();
    session.endSession();
    
    res.status(500).json({
      success: false,
      message: "Failed to subscribe to plan",
      error: error.message,
    });
  }
};

/**
 * Upgrade a user's subscription to a higher-tier plan
 */
export const upgradePlan = async (req, res) => {
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const { newPlanId, paymentMethod } = req.body;
    const userId = req.user._id;

    // Find current active subscription
    const currentSubscription = await Subscription.findOne({
      user: userId,
      status: "active",
    }).populate("plan");

    if (!currentSubscription) {
      await session.abortTransaction();
      session.endSession();
      return res.status(404).json({
        success: false,
        message: "No active subscription found",
      });
    }

    // Get new plan details
    const newPlan = await Plan.findById(newPlanId);
    if (!newPlan || newPlan.status !== "active") {
      await session.abortTransaction();
      session.endSession();
      return res.status(404).json({
        success: false,
        message: "New plan not found or not available",
      });
    }

    // Check if upgrade is allowed
    const currentPlan = currentSubscription.plan;
    if (currentPlan.tier === newPlan.tier || !currentPlan.allowUpgrade) {
      await session.abortTransaction();
      session.endSession();
      return res.status(400).json({
        success: false,
        message: "Upgrade not allowed for your current plan",
      });
    }

    const currentEndDate = currentSubscription.endDate;
    const today = new Date();
    const remainingDays = Math.ceil((currentEndDate - today) / (1000 * 60 * 60 * 24));
    const totalDays = Math.ceil((currentSubscription.endDate - currentSubscription.startDate) / (1000 * 60 * 60 * 24));
    const proratedRefund = (currentPlan.price * remainingDays) / totalDays;
    
    const newEndDate = currentSubscription.endDate;
    
    const upgradeCost = (newPlan.price * remainingDays) / totalDays;
    const priceDifference = upgradeCost - proratedRefund;

    // Update current subscription status to cancelled
    await Subscription.findByIdAndUpdate(
      currentSubscription._id,
      { status: "cancelled" },
      { session }
    );

    // Create new subscription
    const newSubscription = await Subscription.create(
      [{
        user: userId,
        plan: newPlanId,
        startDate: today,
        endDate: newEndDate,
        status: "active",
      }],
      { session }
    );

    // Create billing record for upgrade
    const billing = await Billing.create(
      [{
        user: userId,
        subscription: newSubscription[0]._id,
        amount: priceDifference > 0 ? priceDifference : 0,
        subtotal: priceDifference > 0 ? priceDifference : 0,
        dueDate: new Date(),
        billingPeriod: {
          start: today,
          end: newEndDate,
        },
        paymentMethod: {
          type: paymentMethod.type,
          details: paymentMethod.details,
        },
        items: [
          {
            description: `Upgrade from ${currentPlan.name} to ${newPlan.name} plan`,
            amount: priceDifference > 0 ? priceDifference : 0,
            quantity: 1,
          },
        ],
      }],
      { session }
    );

    // Log the upgrade action
    await createLog({
      user: userId,
      action: "UPGRADE",
      entityType: "SUBSCRIPTION",
      entityId: newSubscription[0]._id,
      details: {
        oldPlanId: currentPlan._id,
        oldPlanName: currentPlan.name,
        newPlanId: newPlan._id,
        newPlanName: newPlan.name,
        proratedAmount: priceDifference,
      },
    }, session);

    await session.commitTransaction();
    session.endSession();

    res.status(200).json({
      success: true,
      message: "Successfully upgraded plan",
      data: {
        subscription: newSubscription[0],
        billing: billing[0],
        proratedAmount: priceDifference,
      },
    });
  } catch (error) {
    await session.abortTransaction();
    session.endSession();
    
    res.status(500).json({
      success: false,
      message: "Failed to upgrade plan",
      error: error.message,
    });
  }
};

/**
 * Downgrade a user's subscription to a lower-tier plan
 */
export const downgradePlan = async (req, res) => {
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const { newPlanId } = req.body;
    const userId = req.user._id;

    // Find current active subscription
    const currentSubscription = await Subscription.findOne({
      user: userId,
      status: "active",
    }).populate("plan");

    if (!currentSubscription) {
      await session.abortTransaction();
      session.endSession();
      return res.status(404).json({
        success: false,
        message: "No active subscription found",
      });
    }

    // Get new plan details
    const newPlan = await Plan.findById(newPlanId);
    if (!newPlan || newPlan.status !== "active") {
      await session.abortTransaction();
      session.endSession();
      return res.status(404).json({
        success: false,
        message: "New plan not found or not available",
      });
    }

    // Check if downgrade is allowed
    const currentPlan = currentSubscription.plan;
    if (currentPlan.tier === newPlan.tier || !currentPlan.allowDowngrade) {
      await session.abortTransaction();
      session.endSession();
      return res.status(400).json({
        success: false,
        message: "Downgrade not allowed for your current plan",
      });
    }
    
    // Most plans don't offer refunds for downgrades, but implement the business logic here
    // For this example, we'll just schedule the downgrade for the next billing cycle
    
    // Calculate new start and end dates (start at current end date = next billing cycle)
    const newStartDate = currentSubscription.endDate;
    const newEndDate = new Date(newStartDate);

    switch (newPlan.billingCycle) {
      case "monthly":
        newEndDate.setMonth(newEndDate.getMonth() + 1);
        break;
      case "quarterly":
        newEndDate.setMonth(newEndDate.getMonth() + 3);
        break;
      case "semi-annual":
        newEndDate.setMonth(newEndDate.getMonth() + 6);
        break;
      case "annual":
        newEndDate.setFullYear(newEndDate.getFullYear() + 1);
        break;
      default:
        newEndDate.setMonth(newEndDate.getMonth() + 1);
    }

    // Log the downgrade action - it will take effect at next billing cycle
    await createLog({
      user: userId,
      action: "DOWNGRADE_SCHEDULED",
      entityType: "SUBSCRIPTION",
      entityId: currentSubscription._id,
      details: {
        currentPlanId: currentPlan._id,
        currentPlanName: currentPlan.name,
        newPlanId: newPlan._id,
        newPlanName: newPlan.name,
        effectiveDate: newStartDate,
      },
    }, session);

    // Create a scheduled task or flag to handle the downgrade when the current subscription expires
    // This implementation depends on your task scheduling system
    // For simplicity, we'll just add a field to the current subscription to indicate pending downgrade
    
    await Subscription.findByIdAndUpdate(
      currentSubscription._id,
      { 
        pendingChange: {
          action: "DOWNGRADE",
          newPlanId: newPlanId,
          effectiveDate: newStartDate
        }
      },
      { session }
    );

    await session.commitTransaction();
    session.endSession();

    res.status(200).json({
      success: true,
      message: "Downgrade scheduled for next billing cycle",
      data: {
        currentSubscription: currentSubscription,
        newPlan: newPlan,
        effectiveDate: newStartDate,
      },
    });
  } catch (error) {
    await session.abortTransaction();
    session.endSession();
    
    res.status(500).json({
      success: false,
      message: "Failed to schedule plan downgrade",
      error: error.message,
    });
  }
};

/**
 * Cancel a user's subscription
 */
export const cancelSubscription = async (req, res) => {
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const { cancellationReason } = req.body;
    const userId = req.user._id;

    // Find current active subscription
    const subscription = await Subscription.findOne({
      user: userId,
      status: "active",
    }).populate("plan");

    if (!subscription) {
      await session.abortTransaction();
      session.endSession();
      return res.status(404).json({
        success: false,
        message: "No active subscription found",
      });
    }

    const plan = subscription.plan;
    const today = new Date();
    
    // Check if there's an early termination fee
    let terminationFee = 0;
    if (plan.earlyTerminationFee > 0) {
      const remainingDays = Math.ceil((subscription.endDate - today) / (1000 * 60 * 60 * 24));
      const totalDays = Math.ceil((subscription.endDate - subscription.startDate) / (1000 * 60 * 60 * 24));
      
      // Only charge termination fee if significant time remains
      if (remainingDays > totalDays * 0.25) { // More than 25% of subscription period remains
        terminationFee = plan.earlyTerminationFee;
      }
    }

    // Update subscription status to cancelled and set end date
    await Subscription.findByIdAndUpdate(
      subscription._id,
      {
        status: "cancelled",
        cancellationDate: today,
        cancellationReason: cancellationReason || "User initiated cancellation",
      },
      { session }
    );

    // If there's a termination fee, create a billing record for it
    if (terminationFee > 0) {
      await Billing.create(
        [{
          user: userId,
          subscription: subscription._id,
          amount: terminationFee,
          subtotal: terminationFee,
          dueDate: today,
          billingPeriod: {
            start: today,
            end: today,
          },
          status: "pending",
          items: [
            {
              description: "Early termination fee",
              amount: terminationFee,
              quantity: 1,
            },
          ],
        }],
        { session }
      );
    }

    // Log the cancellation
    await createLog({
      user: userId,
      action: "CANCEL",
      entityType: "SUBSCRIPTION",
      entityId: subscription._id,
      details: {
        planId: plan._id,
        planName: plan.name,
        cancellationDate: today,
        reason: cancellationReason,
        terminationFee,
      },
    }, session);

    await session.commitTransaction();
    session.endSession();

    res.status(200).json({
      success: true,
      message: "Subscription successfully cancelled",
      data: {
        subscription,
        terminationFee,
      },
    });
  } catch (error) {
    await session.abortTransaction();
    session.endSession();
    
    res.status(500).json({
      success: false,
      message: "Failed to cancel subscription",
      error: error.message,
    });
  }
};

/**
 * Renew a user's expired or about-to-expire subscription
 */
export const renewSubscription = async (req, res) => {
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const { paymentMethod } = req.body;
    const userId = req.user._id;

    // Find current subscription that's expired or about to expire
    const subscription = await Subscription.findOne({
      user: userId,
      $or: [
        { status: "expired" },
        { 
          status: "active",
          endDate: { $lte: new Date(new Date().getTime() + 7 * 24 * 60 * 60 * 1000) } // Within 7 days of expiry
        }
      ]
    }).populate("plan");

    if (!subscription) {
      await session.abortTransaction();
      session.endSession();
      return res.status(404).json({
        success: false,
        message: "No subscription found that needs renewal",
      });
    }

    const plan = subscription.plan;
    
    // Calculate new start and end dates
    const newStartDate = subscription.status === "expired" ? new Date() : subscription.endDate;
    const newEndDate = new Date(newStartDate);

    switch (plan.billingCycle) {
      case "monthly":
        newEndDate.setMonth(newEndDate.getMonth() + 1);
        break;
      case "quarterly":
        newEndDate.setMonth(newEndDate.getMonth() + 3);
        break;
      case "semi-annual":
        newEndDate.setMonth(newEndDate.getMonth() + 6);
        break;
      case "annual":
        newEndDate.setFullYear(newEndDate.getFullYear() + 1);
        break;
      default:
        newEndDate.setMonth(newEndDate.getMonth() + 1);
    }

    // If there's a pending plan change (e.g., downgrade), apply it during renewal
    let newPlanId = plan._id;
    let renewalPlan = plan;
    
    if (subscription.pendingChange && subscription.pendingChange.action === "DOWNGRADE") {
      newPlanId = subscription.pendingChange.newPlanId;
      renewalPlan = await Plan.findById(newPlanId);
      
      if (!renewalPlan) {
        newPlanId = plan._id;
        renewalPlan = plan;
      }
    }

    // For expired subscriptions, create a new one
    // For active subscriptions, update the existing one
    let renewedSubscription;
    
    if (subscription.status === "expired") {
      renewedSubscription = await Subscription.create(
        [{
          user: userId,
          plan: newPlanId,
          startDate: newStartDate,
          endDate: newEndDate,
          status: "active",
        }],
        { session }
      );
      
      renewedSubscription = renewedSubscription[0];
    } else {
      await Subscription.findByIdAndUpdate(
        subscription._id,
        {
          plan: newPlanId,
          endDate: newEndDate,
          pendingChange: null, // Clear any pending changes
        },
        { session, new: true }
      );
      
      renewedSubscription = subscription;
    }

    // Create billing record
    const billing = await Billing.create(
      [{
        user: userId,
        subscription: renewedSubscription._id,
        amount: renewalPlan.price,
        subtotal: renewalPlan.price,
        dueDate: new Date(),
        billingPeriod: {
          start: newStartDate,
          end: newEndDate,
        },
        paymentMethod: {
          type: paymentMethod.type,
          details: paymentMethod.details,
        },
        items: [
          {
            description: `Renewal of ${renewalPlan.name} plan`,
            amount: renewalPlan.price,
            quantity: 1,
          },
        ],
      }],
      { session }
    );

    // Log the renewal
    await createLog({
      user: userId,
      action: "RENEW",
      entityType: "SUBSCRIPTION",
      entityId: renewedSubscription._id,
      details: {
        planId: renewalPlan._id,
        planName: renewalPlan.name,
        startDate: newStartDate,
        endDate: newEndDate,
        planChanged: newPlanId.toString() !== plan._id.toString(),
      },
    }, session);

    await session.commitTransaction();
    session.endSession();

    res.status(200).json({
      success: true,
      message: "Subscription successfully renewed",
      data: {
        subscription: renewedSubscription,
        billing: billing[0],
      },
    });
  } catch (error) {
    await session.abortTransaction();
    session.endSession();
    
    res.status(500).json({
      success: false,
      message: "Failed to renew subscription",
      error: error.message,
    });
  }
};

/**
 * Get a user's subscription history
 */
export const getSubscriptionHistory = async (req, res) => {
  try {
    const userId = req.user._id;
    
    const subscriptions = await Subscription.find({ user: userId })
      .populate("plan", "name price tier features")
      .sort({ createdAt: -1 });
    
    const billings = await Billing.find({ user: userId })
      .sort({ createdAt: -1 });
      
    res.status(200).json({
      success: true,
      data: {
        subscriptions,
        billings,
      },
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "Failed to fetch subscription history",
      error: error.message,
    });
  }
};

/**
 * Get details of user's active subscription
 */
export const getActiveSubscription = async (req, res) => {
  try {
    const userId = req.user._id;
    
    const subscription = await Subscription.findOne({
      user: userId,
      status: "active",
    }).populate("plan");
    
    if (!subscription) {
      return res.status(404).json({
        success: false,
        message: "No active subscription found",
      });
    }
    
    // Check if the subscription has expired but hasn't been updated
    subscription.checkStatus();
    
    // If it's now expired, save the changes and return appropriate response
    if (subscription.status === "expired") {
      await subscription.save();
      
      return res.status(200).json({
        success: true,
        message: "Your subscription has expired",
        data: {
          subscription,
          isActive: false,
        },
      });
    }
    
    res.status(200).json({
      success: true,
      data: {
        subscription,
        isActive: true,
      },
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "Failed to fetch active subscription",
      error: error.message,
    });
  }
};
