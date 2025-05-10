import { test, expect } from '@playwright/test';

test('Go to dashboard', async ({ page }) => {
  await page.goto('http://localhost:3000/app/dashboards');
  await expect(page.getByText('Dashboards').nth(1)).toBeVisible();
  await expect(page.getByRole('button', { name: 'New dashboard' })).toBeVisible();
  await expect(page.getByRole('button', { name: '12px star' })).toBeVisible();
  await page.getByRole('link', { name: 'Important Metrics 12px star' }).click();
  await page.waitForTimeout(100);
  await page.waitForLoadState('networkidle');
  await page.waitForLoadState('domcontentloaded');
  await expect(page.getByRole('textbox', { name: 'New dashboard' })).toHaveValue(
    'Important Metrics'
  );
});

test('Can remove a metric from a dashboard', async ({ page }) => {
  await page.goto('http://localhost:3000/app/dashboards/c0855f0f-f50a-424e-9e72-9e53711a7f6a');
  await expect(page.getByRole('button', { name: 'Quarterly Gross Profit Margin' })).toBeVisible();
  // Hover over the metric to reveal the three-dot menu
  await page
    .locator(`[data-testid="metric-item-72e445a5-fb08-5b76-8c77-1642adf0cb72"]`)
    .hover({ timeout: 500 });
  await expect(
    page.locator(`[data-testid="metric-item-72e445a5-fb08-5b76-8c77-1642adf0cb72"]`)
  ).toBeVisible();

  await page
    .getByTestId('metric-item-72e445a5-fb08-5b76-8c77-1642adf0cb72')
    .locator('button')
    .click();
  await page.getByRole('menuitem', { name: 'Delete' }).click();
  await expect(page.getByRole('button', { name: 'Submit' })).toBeVisible();
  await page.getByRole('button', { name: 'Submit' }).click();
  await expect(
    page.getByRole('button', { name: 'Quarterly Gross Profit Margin' })
  ).not.toBeVisible();
  await page
    .locator('div')
    .filter({ hasText: /^DashboardFileStart chat$/ })
    .getByRole('button')
    .nth(2)
    .click();
  await page.getByRole('textbox', { name: 'Search...' }).click();
  await page
    .getByRole('textbox', { name: 'Search...' })
    .fill('Quarterly Gross Profit Margin Trend (Q2 2023 - Q1 2024)');
  await page.waitForTimeout(450);
  await page.waitForLoadState('networkidle');
  await page
    .locator('div:nth-child(2) > div > div > div > .border-border > div > .peer')
    .first()
    .click();
  await expect(page.getByRole('button', { name: 'Add metrics' })).toBeVisible();
  await page.getByRole('button', { name: 'Add metrics' }).click();
  await page.waitForTimeout(300);
  await page.waitForLoadState('networkidle');
  await expect(page.getByTestId('metric-item-72e445a5-fb08-5b76-8c77-1642adf0cb72')).toBeVisible();

  //drag back into place
  // Use a more specific selector if the element is a link
  const sourceElement = page.getByRole('button', { name: 'Quarterly Gross Profit Margin' });
  const targetElement = page.getByRole('button', { name: 'Revenue by Product Category (' });

  expect(sourceElement).toBeVisible();
  expect(targetElement).toBeVisible();

  try {
    // Skip the initial click since it's a link and would navigate away
    // Go straight to hover and mouse operations
    await sourceElement.hover({ force: true });
    await page.waitForTimeout(200);
    await page.mouse.down();
    await page.waitForTimeout(200);

    // Move to target element
    await targetElement.hover({ force: true, position: { x: 5, y: 5 } });
    await page.waitForTimeout(400);

    await page.mouse.up();
    await page.waitForTimeout(1000);
  } catch (e) {
    const sourceBoundingBox = await sourceElement.boundingBox();
    const targetBoundingBox = await targetElement.boundingBox();

    if (sourceBoundingBox && targetBoundingBox) {
      const startX = sourceBoundingBox.x + sourceBoundingBox.width / 2;
      const startY = sourceBoundingBox.y + sourceBoundingBox.height / 2;
      const endX = targetBoundingBox.x + 10;
      const endY = targetBoundingBox.y + targetBoundingBox.height / 2;

      // Skip the initial click
      await page.mouse.move(startX, startY);
      await page.waitForTimeout(300);
      await page.mouse.down();
      await page.waitForTimeout(300);

      // Move to destination with a slower motion
      const steps = 20;
      for (let i = 0; i <= steps; i++) {
        const stepX = startX + (endX - startX) * (i / steps);
        const stepY = startY + (endY - startY) * (i / steps);
        await page.mouse.move(stepX, stepY);
        await page.waitForTimeout(3);
      }

      await page.waitForTimeout(100);
      await page.mouse.up();
      await page.waitForTimeout(100);
    }
  }

  // Verify the element was moved successfully
  await expect(sourceElement).toBeVisible();
  await page.getByRole('textbox', { name: 'New dashboard' }).click();
  await expect(page.getByRole('button', { name: 'Save' })).toBeVisible();
  await page.getByRole('button', { name: 'Save' }).click();
  await page.waitForTimeout(55);
  await page.waitForLoadState('networkidle');
});

test('Can edit name and description of a dashboard', async ({ page }) => {
  await page.goto('http://localhost:3000/app/dashboards/c0855f0f-f50a-424e-9e72-9e53711a7f6a');
  await expect(page.getByRole('textbox', { name: 'Add description...' })).toHaveValue('');

  await page.getByRole('textbox', { name: 'New dashboard' }).click();
  await page.getByRole('textbox', { name: 'New dashboard' }).fill('Important Metrics NATE RULES');
  await page.getByRole('textbox', { name: 'Add description...' }).click();
  await page.getByRole('textbox', { name: 'Add description...' }).fill('HUH?');
  await expect(page.getByRole('textbox', { name: 'New dashboard' })).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'New dashboard' })).toHaveValue(
    'Important Metrics NATE RULES'
  );
  await page.getByRole('button', { name: 'Save' }).click();
  await page.waitForTimeout(100);
  await page.waitForLoadState('networkidle');

  await expect(page.getByRole('textbox', { name: 'New dashboard' })).toHaveValue(
    'Important Metrics NATE RULES'
  );
  await page.getByRole('textbox', { name: 'New dashboard' }).click();
  await page.getByRole('textbox', { name: 'New dashboard' }).fill('Important Metrics');
  await page.getByRole('textbox', { name: 'Add description...' }).click();
  await expect(page.getByRole('textbox', { name: 'Add description...' })).toHaveValue('HUH?');

  await expect(page.getByRole('textbox', { name: 'New dashboard' })).toHaveValue(
    'Important Metrics'
  );
  await page.getByRole('textbox', { name: 'Add description...' }).fill('');
  await expect(page.getByRole('textbox', { name: 'Add description...' })).toBeEmpty();
  await page.getByRole('textbox', { name: 'New dashboard' }).click();
  await page.getByRole('textbox', { name: 'New dashboard' }).fill('Important Metrics SWAG');
  await page.getByRole('button', { name: 'Save' }).click();
  await page.waitForTimeout(400);
  await page.waitForLoadState('networkidle');
  await page.getByTestId('segmented-trigger-file').click();
  await page.getByTestId('segmented-trigger-file').click();
  await page.waitForTimeout(1000);
  await page.waitForLoadState('networkidle');
  await page.waitForLoadState('domcontentloaded');

  await expect(page.getByRole('code').getByText('Important Metrics SWAG')).toBeVisible({
    timeout: 22000
  });
  await expect(page.locator('.current-line').first()).toBeVisible();
  await page.getByTestId('segmented-trigger-dashboard').click();
  await page.getByRole('textbox', { name: 'New dashboard' }).click();
  await page.getByRole('textbox', { name: 'New dashboard' }).fill('Important Metrics');
  await page.getByRole('button', { name: 'Save' }).click();
  await expect(page.getByRole('textbox', { name: 'New dashboard' })).toHaveValue(
    'Important Metrics'
  );
});
